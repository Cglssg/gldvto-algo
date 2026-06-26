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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_bound=1.0):
        super(Actor, self).__init__()
        self.action_bound = action_bound

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        action = torch.tanh(self.fc4(x)) * self.action_bound
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc4 = nn.Linear(hidden_dim, 1)

        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.relu(self.q1_fc1(sa))
        q1 = torch.relu(self.q1_fc2(q1))
        q1 = torch.relu(self.q1_fc3(q1))
        q1 = self.q1_fc4(q1)

        q2 = torch.relu(self.q2_fc1(sa))
        q2 = torch.relu(self.q2_fc2(q2))
        q2 = torch.relu(self.q2_fc3(q2))
        q2 = self.q2_fc4(q2)

        return q1, q2

    def q1_forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.q1_fc1(sa))
        q1 = torch.relu(self.q1_fc2(q1))
        q1 = torch.relu(self.q1_fc3(q1))
        q1 = self.q1_fc4(q1)
        return q1

class TD3Agent:
    def __init__(self, state_dim, action_dim,
                 learning_rate=3e-4, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, weight_decay=1e-5)

        self.memory = deque(maxlen=100000)
        self.batch_size = 256
        self.warmup_steps = 1000
        self.train_step = 0

        self.exploration_noise = 0.1
        self.action_bound = 1.0

        self.actor_losses = deque(maxlen=10000)
        self.critic_losses = deque(maxlen=10000)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]

        if not evaluate:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = (action + noise).clip(-self.action_bound, self.action_bound)

        action_idx = np.argmax(action) if len(action) > 0 else 0
        return action_idx

    def replay(self):
        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            noise = torch.clamp(
                torch.normal(0, self.policy_noise, size=next_actions.shape).to(self.device),
                -self.noise_clip, self.noise_clip
            )
            next_actions = torch.clamp(next_actions + noise, -self.action_bound, self.action_bound)

            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones.float()) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)

        critic_loss = F.smooth_l1_loss(current_q1, target_q) + F.smooth_l1_loss(current_q2, target_q)
        self.critic_losses.append(critic_loss.item())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        if self.train_step % self.policy_freq == 0:
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
            self.actor_losses.append(actor_loss.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            self._soft_update(self.critic_target, self.critic, self.tau)
            self._soft_update(self.actor_target, self.actor, self.tau)

        self.train_step += 1

    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save_model(self, filepath):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'actor_target_state_dict': self.actor_target.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'critic_target_state_dict': self.critic_target.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'actor_losses': list(self.actor_losses),
                'critic_losses': list(self.critic_losses)
            }, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, filepath):
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.actor_losses = deque(checkpoint.get('actor_losses', []), maxlen=10000)
            self.critic_losses = deque(checkpoint.get('critic_losses', []), maxlen=10000)
            logger.info(f"Model loaded from {filepath}")
        except FileNotFoundError:
            logger.error(f"Model file {filepath} not found")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def reset_losses(self):
        self.actor_losses.clear()
        self.critic_losses.clear()

class BaseEnv:
    def __init__(self, num_vehicles=EnvConfig.NUM_VEHICLES,
                 num_base_stations=EnvConfig.NUM_BASESTATIONS,
                 num_edge_servers=EnvConfig.NUM_EDGE_SERVERS):
        self.num_vehicles = num_vehicles
        self.num_base_stations = num_base_stations
        self.num_edge_servers = num_edge_servers

        self.current_state = None

        self.simulator = Simulator(calculate_reward_fn = self._calculate_reward)

        self.simulator.setup_scenario(num_vehicles, num_base_stations, num_edge_servers)
        self.metrics = self.simulator.metrics
        self.metrics_episodes = []

    def reset(self):
        self.metrics_episodes.append(copy.deepcopy(self.metrics))

        self.simulator = Simulator(calculate_reward_fn = self._calculate_reward)
        self.simulator.setup_scenario(self.num_vehicles, self.num_base_stations, self.num_edge_servers)
        self.metrics = self.simulator.metrics
        self.current_state = None

    def normalize_state(self, state):
        state = np.array(state, dtype=np.float32)
        vehicle_dim = self.num_vehicles * 4
        bs_dim = vehicle_dim + self.num_base_stations * 2
        edge_dim = bs_dim + self.num_edge_servers * 4

        if vehicle_dim > 0:
            vehicle_state = state[:vehicle_dim]
            max_v = np.max(np.abs(vehicle_state))
            if max_v > 0:
                state[:vehicle_dim] = vehicle_state / max_v

        if bs_dim > vehicle_dim:
            bs_state = state[vehicle_dim:bs_dim]
            max_v = np.max(np.abs(bs_state))
            if max_v > 0:
                state[vehicle_dim:bs_dim] = bs_state / max_v

        if edge_dim > bs_dim:
            edge_state = state[bs_dim:edge_dim]
            max_v = np.max(np.abs(edge_state))
            if max_v > 0:
                state[bs_dim:edge_dim] = edge_state / max_v

        cloud_state = state[edge_dim:]
        max_v = np.max(np.abs(cloud_state))
        if max_v > 0:
            state[edge_dim:] = cloud_state / max_v

        return state

    def get_state(self):
        state_features = []
        for vehicle in self.simulator.vehicles:
            cpu_util = vehicle.cpu_usage / vehicle.cpu_capacity if hasattr(vehicle,
                                                                           'cpu_usage') and vehicle.cpu_capacity > 0 else 0.0
            mem_util = vehicle.memory_usage / vehicle.memory_capacity if hasattr(vehicle,
                                                                                 'memory_usage') and vehicle.memory_capacity > 0 else 0.0
            state_features.extend([
                vehicle.position[0] / 12.0,
                vehicle.position[1] / 12.0,
                np.clip(cpu_util, 0, 1),
                np.clip(mem_util, 0, 1)
            ])

        for bs in self.simulator.base_stations:
            bandwidth_norm = bs.bandwidth / 5000.0
            connected_norm = len(bs.connected_vehicles) / self.num_vehicles if self.num_vehicles > 0 else 0.0
            state_features.extend([
                np.clip(bandwidth_norm, 0, 1),
                np.clip(connected_norm, 0, 1)
            ])

        for es in self.simulator.edge_servers:
            cpu_util = es.cpu_usage / es.cpu_capacity if hasattr(es, 'cpu_usage') and es.cpu_capacity > 0 else 0.0
            mem_util = es.memory_usage / es.memory_capacity if hasattr(es,
                                                                       'memory_usage') and es.memory_capacity > 0 else 0.0
            state_features.extend([
                np.clip(cpu_util, 0, 1),
                np.clip(mem_util, 0, 1),
                np.clip(es.current_load, 0, 1),
                es.energy_consumption / 1000.0
            ])

        cloud = self.simulator.cloud_server
        state_features.extend([
            cloud.compute_capacity / 1000.0,
            cloud.communication_delay / 0.1,
            cloud.energy_consumption / 1000.0
        ])

        target_length = self.num_vehicles * 4 + self.num_base_stations * 2 + self.num_edge_servers * 4 + 3
        if len(state_features) < target_length:
            state_features += [0.0] * (target_length - len(state_features))
        else:
            state_features = state_features[:target_length]
        return np.array(state_features, dtype=np.float32)

    def step(self,action):
        reward = self.simulator.run_step(action)

        next_state = self.normalize_state(self.get_state())
        self.current_state = next_state

        done = self.simulator.current_time >= EnvConfig.STEP

        return next_state, reward, done

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

class TD3Env(BaseEnv):
    def step(self, action_idx):
        action = np.zeros(self.num_edge_servers + 1)
        action[action_idx] = 1.0

        reward = self.simulator.run_step(action_idx)
        next_state = self.normalize_state(self.get_state())
        self.current_state = next_state
        done = self.simulator.current_time >= EnvConfig.STEP

        return next_state, reward, done

def train_td3(lr=1e-5, gamma=0.8, tau=0.005, policy_noise=0.5, episodes_num=EnvConfig.EPISODES):
    random.seed(EnvConfig.RANDOM_SEED)
    np.random.seed(EnvConfig.RANDOM_SEED)
    torch.manual_seed(EnvConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EnvConfig.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env = TD3Env(
        num_vehicles=EnvConfig.NUM_VEHICLES,
        num_base_stations=EnvConfig.NUM_BASESTATIONS,
        num_edge_servers=EnvConfig.NUM_EDGE_SERVERS
    )

    state_dim = env.num_vehicles * 4 + env.num_base_stations * 2 + env.num_edge_servers * 4 + 3
    action_dim = 3

    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=lr,
        gamma=gamma,
        tau=tau,
        policy_noise=policy_noise
    )

    rewards_history = []
    actor_losses_history = []
    critic_losses_history = []
    logger.info(f"Start training TD3 with lr={lr}, gamma={gamma}, tau={tau}, policy_noise={policy_noise}")

    for episode in tqdm(range(episodes_num), desc="Training TD3", unit="episode"):
        state = env.normalize_state(env.get_state())
        total_reward = 0
        done = False
        episode_actor_losses = []
        episode_critic_losses = []

        while not done:
            action_idx = agent.act(state)
            next_state, reward, done = env.step(action_idx)
            action = np.zeros(action_dim)
            action[action_idx] = 1.0
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

            if agent.actor_losses:
                episode_actor_losses.append(agent.actor_losses[-1])
            if agent.critic_losses:
                episode_critic_losses.append(agent.critic_losses[-1])

        rewards_history.append(total_reward)
        actor_losses_history.append(np.mean(episode_actor_losses) if episode_actor_losses else 0)
        critic_losses_history.append(np.mean(episode_critic_losses) if episode_critic_losses else 0)

        env.reset()

    return rewards_history, env.metrics_episodes, (lr, gamma, tau, policy_noise)