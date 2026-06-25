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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义根目录和子目录
ROOT_RESULT_DIR = "result_gcn_d3qn"
MODEL_DIR = os.path.join(ROOT_RESULT_DIR, "model")
PLOT_DIR = os.path.join(ROOT_RESULT_DIR, "plt")

# 确保目录存在
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


class D3QN(nn.Module):
    """GCN+D3QN网络：Dueling架构（分离价值流和优势流）"""

    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=128, action_dim=3):
        super(D3QN, self).__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # GCN层：提取图结构特征（与原DDQN一致）
        self.conv1 = pyg_nn.GCNConv(node_feature_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.conv3 = pyg_nn.GCNConv(hidden_dim, hidden_dim // 2)

        # Dueling架构：价值流（V）和优势流（A）
        # 价值流：输出状态价值（标量）
        self.value_fc1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.value_fc2 = nn.Linear(hidden_dim, 1)

        # 优势流：输出每个动作的优势值（动作维度）
        self.advantage_fc1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.advantage_fc2 = nn.Linear(hidden_dim, action_dim)

        # 激活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        """
        前向传播：Dueling架构计算Q值
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        :param data: 包含x(节点特征), edge_index(边特征)
        :return: Q值
        """
        x, edge_index = data.x, data.edge_index

        # GCN特征提取（与原DDQN一致）
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv3(x, edge_index))

        # 全局平均池化
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        x_pooled = pyg_nn.global_mean_pool(x, batch)

        # 价值流计算
        value = self.relu(self.value_fc1(x_pooled))
        value = self.value_fc2(value)  # [batch_size, 1]

        # 优势流计算
        advantage = self.relu(self.advantage_fc1(x_pooled))
        advantage = self.advantage_fc2(advantage)  # [batch_size, action_dim]

        # 组合Q值（消除优势流的尺度歧义）
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class D3QNAgent:
    """集成GCN的D3QN智能体（Dueling Double DQN）"""

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

        # 设备自动选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 主网络和目标网络（GCN+D3QN）
        self.policy_net = D3QN(node_feature_dim, edge_feature_dim, action_dim=action_dim).to(self.device)
        self.target_net = D3QN(node_feature_dim, edge_feature_dim, action_dim=action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器（与原DDQN一致）
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)

        # 经验回放缓冲区（与原DDQN一致）
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        # 训练控制（与原DDQN一致）
        self.train_step = 0
        self.target_update = 50
        self.losses = deque(maxlen=10000)

    def remember(self, graph_data, action, reward, next_graph_data, done):
        """存储图结构经验（与原DDQN一致）"""
        self.memory.append((graph_data, action, reward, next_graph_data, done))

    def act(self, graph_data, evaluate=False):
        """基于图特征选择动作（与原DDQN一致）"""
        if not evaluate and np.random.random() <= self.epsilon:
            return random.randrange(self.action_dim)

        # 转换为batch格式并计算Q值
        graph_data = graph_data.to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(graph_data)
        return np.argmax(q_values.cpu().numpy())

    def replay(self):
        """GCN+D3QN经验回放训练（核心改动：DDQN逻辑适配Dueling架构）"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        graph_datas, actions, rewards, next_graph_datas, dones = zip(*batch)

        # 构建批次数据（与原DDQN一致）
        batch_data = Batch.from_data_list([gd.to('cpu') for gd in graph_datas]).to(self.device)
        next_batch_data = Batch.from_data_list([gd.to('cpu') for gd in next_graph_datas]).to(self.device)

        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # 当前Q值（与原DDQN一致）
        current_q_values = self.policy_net(batch_data).gather(1, actions.unsqueeze(1)).squeeze()

        # D3QN核心：DDQN逻辑 + Dueling价值计算
        with torch.no_grad():
            # 策略网络选动作（与原DDQN一致）
            next_q_policy = self.policy_net(next_batch_data)
            best_actions = next_q_policy.argmax(1)

            # 目标网络评估Q值（Dueling架构下自动计算）
            next_q_target = self.target_net(next_batch_data)
            next_q_values = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze()

        # 目标Q值（与原DDQN一致）
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # 计算损失（与原DDQN一致）
        loss = F.mse_loss(current_q_values, target_q_values)
        self.losses.append(loss.item())

        # 反向传播（与原DDQN一致）
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 更新探索率（与原DDQN一致）
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 更新目标网络（与原DDQN一致）
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, filepath):
        """保存模型（与原DDQN一致）"""
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
        """加载模型（与原DDQN一致）"""
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
        """重置loss列表（与原DDQN一致）"""
        self.losses.clear()


class GCN_D3QNEnv:
    """适配GCN的环境：与原DDQN环境完全一致，仅类名调整"""

    def __init__(self, num_vehicles=EnvConfig.NUM_VEHICLES,
                 num_base_stations=EnvConfig.NUM_BASESTATIONS,
                 num_edge_servers=EnvConfig.NUM_EDGE_SERVERS,
                 d3qn_agent=None):
        self.num_vehicles = num_vehicles
        self.num_base_stations = num_base_stations
        self.num_edge_servers = num_edge_servers

        self.d3qn_agent = d3qn_agent
        self.current_graph_data = None

        # 初始化模拟器
        self.simulator = Simulator(
                                   calculate_reward_fn=self._calculate_reward)
        self.simulator.setup_scenario(num_vehicles, num_base_stations, num_edge_servers)
        self.metrics = self.simulator.metrics
        self.metrics_episodes = []

        # 节点特征维度（保持8维不变）
        self.node_feature_dim = 8

    def reset(self):
        """重置环境并生成初始图数据（与原DDQN一致）"""
        self.metrics_episodes.append(copy.deepcopy(self.metrics))

        self.simulator = Simulator(
                                   calculate_reward_fn=self._calculate_reward)
        self.simulator.setup_scenario(self.num_vehicles, self.num_base_stations, self.num_edge_servers)
        self.metrics = self.simulator.metrics
        self.current_graph_data = self.build_graph_data()
        return self.current_graph_data

    def build_graph_data(self):
        """构建图结构数据（与原DDQN完全一致）"""
        # 1. 收集所有节点
        nodes = []
        node_types = []  # 0:车辆, 1:基站, 2:边缘, 3:云端

        # 车辆节点特征（8维）
        for vehicle in self.simulator.vehicles:
            cpu_util = vehicle.cpu_usage / vehicle.cpu_capacity if vehicle.cpu_capacity > 0 else 0.0
            mem_util = vehicle.memory_usage / vehicle.memory_capacity if vehicle.memory_capacity > 0 else 0.0
            pos_x = vehicle.position[0] / 12.0
            pos_y = vehicle.position[1] / 12.0
            task_load = len(vehicle.tasks) / 10.0 if hasattr(vehicle, 'tasks') else 0.0
            battery = vehicle.battery / 100.0 if hasattr(vehicle, 'battery') else 1.0
            delay_sensitivity = 1.0  # 任务延迟敏感度
            priority = 1.0  # 任务优先级

            nodes.append([pos_x, pos_y, cpu_util, mem_util, task_load, battery, delay_sensitivity, priority])
            node_types.append(0)

        # 基站节点特征（8维，不足补0）
        for bs in self.simulator.base_stations:
            bandwidth = bs.bandwidth / 5000.0
            connected_vehicles = len(bs.connected_vehicles) / self.num_vehicles
            pos_x = bs.position[0] / 12.0
            pos_y = bs.position[1] / 12.0
            nodes.append([pos_x, pos_y, bandwidth, connected_vehicles, 0, 0, 0, 0])
            node_types.append(1)

        # 边缘服务器节点特征（8维）
        for es in self.simulator.edge_servers:
            cpu_util = es.cpu_usage / es.cpu_capacity if es.cpu_capacity > 0 else 0.0
            mem_util = es.memory_usage / es.memory_capacity if es.memory_capacity > 0 else 0.0
            load = es.current_load
            energy = es.energy_consumption / 1000.0
            pos_x = es.position[0] / 12.0 if hasattr(es, 'position') else 0.0
            pos_y = es.position[1] / 12.0 if hasattr(es, 'position') else 0.0
            nodes.append([pos_x, pos_y, cpu_util, mem_util, load, energy, 0, 0])
            node_types.append(2)

        # 云端节点特征（8维）
        cloud = self.simulator.cloud_server
        compute_cap = cloud.compute_capacity / 1000.0
        delay = cloud.communication_delay / 0.1
        energy = cloud.energy_consumption / 1000.0
        nodes.append([0.5, 0.5, compute_cap, delay, energy, 0, 0, 0])
        node_types.append(3)

        # 2. 构建边（邻接矩阵）
        edge_index = [[], []]
        edge_attr = []  # 边特征：距离/连接强度

        # 车辆-基站边（基于距离）
        vehicle_idx = 0
        for vehicle in self.simulator.vehicles:
            bs_idx = self.num_vehicles
            for bs in self.simulator.base_stations:
                distance = vehicle.calculate_distance(vehicle.position, bs.position)
                if distance < 3.0:  # 距离小于3则建立连接
                    strength = 1.0 - (distance / 3.0)  # 连接强度（反比于距离）
                    edge_index[0].append(vehicle_idx)
                    edge_index[1].append(bs_idx)
                    edge_attr.append([strength])

                    # 无向图：反向边
                    edge_index[0].append(bs_idx)
                    edge_index[1].append(vehicle_idx)
                    edge_attr.append([strength])
                bs_idx += 1
            vehicle_idx += 1

        # 基站-边缘服务器边
        bs_idx = self.num_vehicles
        for bs in self.simulator.base_stations:
            es_idx = self.num_vehicles + self.num_base_stations
            for es in self.simulator.edge_servers:
                if es in bs.connected_edge_servers:
                    edge_index[0].append(bs_idx)
                    edge_index[1].append(es_idx)
                    edge_attr.append([1.0])  # 固定连接强度

                    # 无向图：反向边
                    edge_index[0].append(es_idx)
                    edge_index[1].append(bs_idx)
                    edge_attr.append([1.0])
                es_idx += 1
            bs_idx += 1

        # 3. 转换为PyG Data对象
        x = torch.FloatTensor(nodes)
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_attr)

        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return graph_data

    def step(self, action):
        """执行动作并返回新的图状态（与原DDQN一致）"""
        reward = self.simulator.run_step(action)
        next_graph_data = self.build_graph_data()
        self.current_graph_data = next_graph_data
        done = self.simulator.current_time >= EnvConfig.STEP

        return next_graph_data, reward, done

    def _calculate_reward(self, success, rtt, delay_constraint, energy, task_priority=1):
        """奖励函数（保持原有逻辑）"""
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
    """训练GCN+D3QN模型（仅代理和环境类名调整）"""
    # 固定随机种子
    random.seed(EnvConfig.RANDOM_SEED)
    np.random.seed(EnvConfig.RANDOM_SEED)
    torch.manual_seed(EnvConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EnvConfig.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 初始化环境
    env = GCN_D3QNEnv(num_vehicles=EnvConfig.NUM_VEHICLES,
                      num_base_stations=EnvConfig.NUM_BASESTATIONS,
                      num_edge_servers=EnvConfig.NUM_EDGE_SERVERS)

    # 初始化智能体（节点特征8维，边特征1维，动作3维）
    agent = D3QNAgent(node_feature_dim=8,
                      edge_feature_dim=1,
                      action_dim=3,
                      learning_rate=lr,
                      gamma=gamma,
                      epsilon=epsilon,
                      epsilon_decay=epsilon_decay)
    env.d3qn_agent = agent

    # 训练记录
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

        # 保存每轮总奖励
        rewards_history.append(total_reward)
        # 保存每轮总损失值
        losses_history.append(np.sum(episode_losses))

    # 模型保存
    model_path = os.path.join(MODEL_DIR, f"gcn_d3qn_lr{lr}_gamma{gamma}_decay{epsilon_decay}.pth")
    agent.save_model(model_path)

    return rewards_history, env.metrics_episodes, (lr, gamma, epsilon_decay)