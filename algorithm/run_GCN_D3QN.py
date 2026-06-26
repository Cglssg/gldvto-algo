import itertools
import copy
from algo_config import EnvConfig
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import matplotlib
from tqdm import tqdm
matplotlib.use('TkAgg')
from env import *
import os
import numpy as np
import random
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class D3QN(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=128, action_dim=3):
        super(D3QN, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.conv1 = pyg_nn.GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.conv3 = pyg_nn.GCNConv(hidden_dim, hidden_dim // 2)

        self.value_fc1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, 1)

        self.advantage_fc1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.advantage_fc2 = nn.Linear(hidden_dim, action_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv3(x, edge_index))

        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        x_pooled = pyg_nn.global_mean_pool(x, batch)

        value = self.relu(self.value_fc1(x_pooled))
        value = self.value_fc2(value)

        advantage = self.relu(self.advantage_fc1(x_pooled))
        advantage = self.advantage_fc2(advantage)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class D3QNAgent:
    def __init__(self, node_feature_dim, edge_feature_dim, action_dim=3,
                 learning_rate=0.00001, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.99):
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = D3QN(node_feature_dim, edge_feature_dim, action_dim=action_dim).to(self.device)
        self.target_net = D3QN(node_feature_dim, edge_feature_dim, action_dim=action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)

        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        self.train_step = 0
        self.target_update = 50
        self.losses = deque(maxlen=10000)

    def remember(self, graph_data, action, reward, next_graph_data, done):
        self.memory.append((graph_data, action, reward, next_graph_data, done))

    def act(self, graph_data, evaluate=False):
        if not evaluate and np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)

        graph_data = graph_data.to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(graph_data)
        return np.argmax(q_values.cpu().numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        graph_datas, actions, rewards, next_graph_datas, dones = zip(*batch)

        batch_data = Batch.from_data_list([gd.to('cpu') for gd in graph_datas]).to(self.device)
        next_batch_data = Batch.from_data_list([gd.to('cpu') for gd in next_graph_datas]).to(self.device)

        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        current_q_values = self.policy_net(batch_data).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q_policy = self.policy_net(next_batch_data)
            best_actions = next_q_policy.argmax(1)

            next_q_target = self.target_net(next_batch_data)
            next_q_values = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze()

        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        loss = F.mse_loss(current_q_values, target_q_values)
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filepath):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'losses': list(self.losses)
            }, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, filepath):
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            self.losses = deque(checkpoint.get('losses', []), maxlen=10000)
            logger.info(f"Model loaded from {filepath}")
        except FileNotFoundError:
            logger.error(f"Model file {filepath} not found")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def reset_losses(self):
        self.losses.clear()


class GCN_D3QNEnv:
    def __init__(self, num_vehicles=EnvConfig.NUM_VEHICLES,
                 num_base_stations=EnvConfig.NUM_BASESTATIONS,
                 num_edge_servers=EnvConfig.NUM_EDGE_SERVERS,
                 d3qn_agent=None):
        self.num_vehicles = num_vehicles
        self.num_base_stations = num_base_stations
        self.num_edge_servers = num_edge_servers

        self.d3qn_agent = d3qn_agent
        self.current_graph_data = None

        self.simulator = Simulator(
                                   calculate_reward_fn=self._calculate_reward)
        self.simulator.setup_scenario(num_vehicles, num_base_stations, num_edge_servers)
        self.metrics = self.simulator.metrics
        self.metrics_episodes = []

        self.node_feature_dim = 8

    def reset(self):
        self.metrics_episodes.append(copy.deepcopy(self.metrics))

        self.simulator = Simulator(
                                   calculate_reward_fn=self._calculate_reward)
        self.simulator.setup_scenario(self.num_vehicles, self.num_base_stations, self.num_edge_servers)
        self.metrics = self.simulator.metrics
        self.current_graph_data = self.build_graph_data()
        return self.current_graph_data

    def build_graph_data(self):
        nodes = []
        node_types = []

        for vehicle in self.simulator.vehicles:
            cpu_util = vehicle.cpu_usage / vehicle.cpu_capacity if vehicle.cpu_capacity > 0 else 0.0
            mem_util = vehicle.memory_usage / vehicle.memory_capacity if vehicle.memory_capacity > 0 else 0.0
            pos_x = vehicle.position[0] / 12.0
            pos_y = vehicle.position[1] / 12.0
            task_load = len(vehicle.tasks) / 10.0 if hasattr(vehicle, 'tasks') else 0.0
            battery = vehicle.battery / 100.0 if hasattr(vehicle, 'battery') else 1.0
            delay_sensitivity = 1.0
            priority = 1.0

            nodes.append([pos_x, pos_y, cpu_util, mem_util, task_load, battery, delay_sensitivity, priority])
            node_types.append(0)

        for bs in self.simulator.base_stations:
            bandwidth = bs.bandwidth / 5000.0
            connected_vehicles = len(bs.connected_vehicles) / self.num_vehicles
            pos_x = bs.position[0] / 12.0
            pos_y = bs.position[1] / 12.0
            nodes.append([pos_x, pos_y, bandwidth, connected_vehicles, 0, 0, 0, 0])
            node_types.append(1)

        for es in self.simulator.edge_servers:
            cpu_util = es.cpu_usage / es.cpu_capacity if es.cpu_capacity > 0 else 0.0
            mem_util = es.memory_usage / es.memory_capacity if es.memory_capacity > 0 else 0.0
            load = es.current_load
            energy = es.energy_consumption / 1000.0
            pos_x = es.position[0] / 12.0 if hasattr(es, 'position') else 0.0
            pos_y = es.position[1] / 12.0 if hasattr(es, 'position') else 0.0
            nodes.append([pos_x, pos_y, cpu_util, mem_util, load, energy, 0, 0])
            node_types.append(2)

        cloud = self.simulator.cloud_server
        compute_cap = cloud.compute_capacity / 1000.0
        delay = cloud.communication_delay / 0.1
        energy = cloud.energy_consumption / 1000.0
        nodes.append([0.5, 0.5, compute_cap, delay, energy, 0, 0, 0])
        node_types.append(3)

        edge_index = [[], []]
        edge_attr = []

        vehicle_idx = 0
        for vehicle in self.simulator.vehicles:
            bs_idx = self.num_vehicles
            for bs in self.simulator.base_stations:
                distance = vehicle.calculate_distance(vehicle.position, bs.position)
                if distance < 3.0:
                    strength = 1.0 - (distance / 3.0)
                    edge_index[0].append(vehicle_idx)
                    edge_index[1].append(bs_idx)
                    edge_attr.append([strength])

                    edge_index[0].append(bs_idx)
                    edge_index[1].append(vehicle_idx)
                    edge_attr.append([strength])
                bs_idx += 1
            vehicle_idx += 1

        bs_idx = self.num_vehicles
        for bs in self.simulator.base_stations:
            es_idx = self.num_vehicles + self.num_base_stations
            for es in self.simulator.edge_servers:
                if es in bs.connected_edge_servers:
                    edge_index[0].append(bs_idx)
                    edge_index[1].append(es_idx)
                    edge_attr.append([1.0])

                    edge_index[0].append(es_idx)
                    edge_index[1].append(bs_idx)
                    edge_attr.append([1.0])
                es_idx += 1
            bs_idx += 1

        x = torch.FloatTensor(nodes)
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_attr)

        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return graph_data

    def step(self, action):
        reward = self.simulator.run_step(action)
        next_graph_data = self.build_graph_data()
        self.current_graph_data = next_graph_data
        done = self.simulator.current_time >= EnvConfig.STEP

        return next_graph_data, reward, done

    def _calculate_reward(self, success, rtt, delay_constraint, energy, task_priority=1):
        task_priority = max(1, min(5, task_priority))
        if not success:
            return -5 * task_priority

        base_reward = 10 * task_priority
        delay_penalty = max(0, min((rtt - delay_constraint) * task_priority, 10 * task_priority))
        delay_reward = max(0, min((delay_constraint - rtt) * 0.5 * task_priority, 5 * task_priority))
        energy_cost = min(energy * 0.1 / task_priority, 5 * task_priority)

        reward = base_reward + delay_reward - delay_penalty - energy_cost
        return max(reward, -5 * task_priority)


def train_gcn_d3qn(lr=4e-5, gamma=0.95, epsilon=1.0, epsilon_decay=0.9, episodes_num=EnvConfig.EPISODES):
    random.seed(EnvConfig.RANDOM_SEED)
    np.random.seed(EnvConfig.RANDOM_SEED)
    torch.manual_seed(EnvConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EnvConfig.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env = GCN_D3QNEnv(num_vehicles=EnvConfig.NUM_VEHICLES,
                      num_base_stations=EnvConfig.NUM_BASESTATIONS,
                      num_edge_servers=EnvConfig.NUM_EDGE_SERVERS)

    agent = D3QNAgent(node_feature_dim=8,
                      edge_feature_dim=1,
                      action_dim=3,
                      learning_rate=lr,
                      gamma=gamma,
                      epsilon=epsilon,
                      epsilon_decay=epsilon_decay)
    env.d3qn_agent = agent

    rewards_history = []
    losses_history = []
    logger.info(f"Start training GCN+D3QN with lr={lr}, gamma={gamma}, decay={epsilon_decay}")

    for episode in tqdm(range(episodes_num), desc="Training GCN+D3QN", unit="episode"):
        graph_data = env.reset()
        total_reward = 0
        done = False
        episode_losses = []

        while not done:
            action = agent.act(graph_data)
            next_graph_data, reward, done = env.step(action)
            agent.remember(graph_data, action, reward, next_graph_data, done)
            agent.replay()

            graph_data = next_graph_data
            total_reward += reward
            if agent.losses:
                episode_losses.append(agent.losses[-1])

        rewards_history.append(total_reward)
        losses_history.append(np.sum(episode_losses))

    return rewards_history, env.metrics_episodes, (lr, gamma, epsilon_decay)