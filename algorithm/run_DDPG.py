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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OU噪声生成器（DDPG探索策略）
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2, sigma_min=0.01, sigma_decay=0.995):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def decay_sigma(self):
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


# Actor网络（策略网络）- 输出确定性动作
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # 输出层使用tanh归一化到[-1,1]，后续映射到离散动作
        action = torch.tanh(self.fc4(x))
        return action


# Critic网络（价值网络）- 评估动作价值
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # 状态分支
        self.fc1 = nn.Linear(state_dim, hidden_dim // 2)
        # 动作分支
        self.fc2 = nn.Linear(action_dim, hidden_dim // 2)
        # 融合层
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        # 状态分支
        s = F.relu(self.fc1(state))
        # 动作分支
        a = F.relu(self.fc2(action))
        # 融合
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # 输出Q值
        q_value = self.fc5(x)
        return q_value


class DDPGAgent:
    """DDPG智能体: Actor-Critic架构"""

    def __init__(self, state_dim, action_dim, lr_actor=0.0001, lr_critic=0.001,
                 gamma=0.9, tau=0.005):  # tau: 软更新系数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau  # 软更新系数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 构建网络
        # 主网络
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        # 目标网络
        self.target_actor = Actor(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)

        # 初始化目标网络权重（与主网络相同）
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-5)

        # 经验回放缓冲区
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        self.warmup_steps = 1000  # 预热步数
        self.replay_interval = 2  # 训练间隔

        # 探索策略
        self.noise = OUNoise(action_dim)
        self.episode = 0

        # 训练参数
        self.train_step = 0
        self.losses_actor = deque(maxlen=10000)
        self.losses_critic = deque(maxlen=10000)

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, evaluate=False):
        """选择动作（评估模式禁用探索）"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Actor输出连续动作
            action = self.actor(state).cpu().numpy()[0]

        # 评估模式直接返回离散动作
        if evaluate:
            return np.argmax(action)

        # 训练模式添加OU噪声探索
        noise = self.noise.sample()
        action = action + noise
        # 归一化并转换为离散动作
        action = np.clip(action, -1, 1)
        return np.argmax(action)

    def replay(self):
        """经验回放训练"""
        # 预热机制
        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return
        # 训练间隔
        if self.train_step % self.replay_interval != 0:
            self.train_step += 1
            return

        # 采样批次
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)  # DDPG动作是连续的
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # ----------------------
        # 训练Critic网络
        # ----------------------
        # 计算目标Q值
        with torch.no_grad():
            # 目标Actor预测下一个状态的动作
            next_actions = self.target_actor(next_states)
            # 目标Critic评估下一个状态-动作对的Q值
            next_q_values = self.target_critic(next_states, next_actions).squeeze()
            # 贝尔曼方程计算目标Q值
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # 主Critic预测当前Q值
        current_q_values = self.critic(states, actions).squeeze()

        # 计算Critic损失（Huber Loss）
        critic_loss = F.smooth_l1_loss(current_q_values, target_q_values)
        self.losses_critic.append(critic_loss.item())

        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # ----------------------
        # 训练Actor网络
        # ----------------------
        # 策略损失：最大化Critic评估的Q值（最小化负Q值）
        current_actions = self.actor(states)
        actor_loss = -self.critic(states, current_actions).mean()
        self.losses_actor.append(actor_loss.item())

        # 优化Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # ----------------------
        # 软更新目标网络
        # ----------------------
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

        self.train_step += 1
        # 衰减噪声
        self.noise.decay_sigma()

    def _soft_update(self, target_net, source_net):
        """软更新目标网络权重"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def save_model(self, filepath):
        """保存模型"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'target_actor_state_dict': self.target_actor.state_dict(),
                'target_critic_state_dict': self.target_critic.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'noise_sigma': self.noise.sigma,
                'losses_actor': list(self.losses_actor),
                'losses_critic': list(self.losses_critic)
            }, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self, filepath):
        """加载模型"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.noise.sigma = checkpoint.get('noise_sigma', 0.2)
            self.losses_actor = deque(checkpoint.get('losses_actor', []), maxlen=10000)
            self.losses_critic = deque(checkpoint.get('losses_critic', []), maxlen=10000)
            logger.info(f"Model loaded from {filepath}")
        except FileNotFoundError:
            logger.error(f"Model file {filepath} not found")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def reset_noise(self):
        """重置噪声"""
        self.noise.reset()


class DDPGEnv:
    """车辆边缘计算环境（DDPG适配版）"""

    def __init__(self, num_vehicles=EnvConfig.NUM_VEHICLES,
                 num_base_stations=EnvConfig.NUM_BASESTATIONS,
                 num_edge_servers=EnvConfig.NUM_EDGE_SERVERS,
                 ddpg_agent=None):
        self.num_vehicles = num_vehicles
        self.num_base_stations = num_base_stations
        self.num_edge_servers = num_edge_servers

        self.ddpg_agent = ddpg_agent  # 保存agent引用
        self.current_state = None

        self.simulator = Simulator(calculate_reward_fn=self._calculate_reward)
        self.simulator.setup_scenario(num_vehicles, num_base_stations, num_edge_servers)
        self.metrics = self.simulator.metrics
        self.metrics_episodes = []

    def reset(self):
        """重置环境"""
        self.metrics_episodes.append(copy.deepcopy(self.metrics))

        self.simulator = Simulator(calculate_reward_fn=self._calculate_reward)
        self.simulator.setup_scenario(self.num_vehicles, self.num_base_stations, self.num_edge_servers)
        self.metrics = self.simulator.metrics
        self.current_state = self.normalize_state(self.get_state())

        # 重置探索噪声
        if self.ddpg_agent:
            self.ddpg_agent.reset_noise()

        return self.current_state

    def normalize_state(self, state):
        """分组归一化：避免单特征主导"""
        state = np.array(state, dtype=np.float32)
        # 按特征分组（车辆4维×N + 基站2维×M + 边缘4维×K + 云端3维）
        vehicle_dim = self.num_vehicles * 4
        bs_dim = vehicle_dim + self.num_base_stations * 2
        edge_dim = bs_dim + self.num_edge_servers * 4

        # 车辆特征归一化
        if vehicle_dim > 0:
            vehicle_state = state[:vehicle_dim]
            max_v = np.max(np.abs(vehicle_state))
            if max_v > 0:
                state[:vehicle_dim] = vehicle_state / max_v

        # 基站特征归一化
        if bs_dim > vehicle_dim:
            bs_state = state[vehicle_dim:bs_dim]
            max_v = np.max(np.abs(bs_state))
            if max_v > 0:
                state[vehicle_dim:bs_dim] = bs_state / max_v

        # 边缘特征归一化
        if edge_dim > bs_dim:
            edge_state = state[bs_dim:edge_dim]
            max_v = np.max(np.abs(edge_state))
            if max_v > 0:
                state[bs_dim:edge_dim] = edge_state / max_v

        # 云端特征归一化
        cloud_state = state[edge_dim:]
        max_v = np.max(np.abs(cloud_state))
        if max_v > 0:
            state[edge_dim:] = cloud_state / max_v

        return state

    def get_state(self):
        """获取环境状态"""
        state_features = []
        # 车辆特征（4维：x, y, cpu利用率, 内存利用率）
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

        # 基站特征（2维：归一化带宽, 归一化连接车辆数）
        for bs in self.simulator.base_stations:
            bandwidth_norm = bs.bandwidth / 5000.0
            connected_norm = len(bs.connected_vehicles) / self.num_vehicles if self.num_vehicles > 0 else 0.0
            state_features.extend([
                np.clip(bandwidth_norm, 0, 1),
                np.clip(connected_norm, 0, 1)
            ])

        # 边缘服务器特征
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

        # 云端特征（3维）
        cloud = self.simulator.cloud_server
        state_features.extend([
            cloud.compute_capacity / 1000.0,
            cloud.communication_delay / 0.1,
            cloud.energy_consumption / 1000.0
        ])

        # 固定维度
        target_length = self.num_vehicles * 4 + self.num_base_stations * 2 + self.num_edge_servers * 4 + 3
        if len(state_features) < target_length:
            state_features += [0.0] * (target_length - len(state_features))
        else:
            state_features = state_features[:target_length]
        return np.array(state_features, dtype=np.float32)

    def step(self, action):
        """执行动作并返回新的状态、奖励和完成标志"""
        # 执行一步仿真
        reward = self.simulator.run_step(action)

        # 获取下一个状态
        next_state = self.normalize_state(self.get_state())
        self.current_state = next_state

        # 判断结束
        done = self.simulator.current_time >= EnvConfig.STEP

        return next_state, reward, done

    def _calculate_reward(self, success, rtt, delay_constraint, energy, task_priority=1):
        """奖励函数（与DQN保持一致）"""
        task_priority = max(1, min(5, task_priority))
        if not success:
            return -5 * task_priority

        # 基础奖励
        base_reward = 10 * task_priority
        # 延迟惩罚（限制最大惩罚）
        delay_penalty = max(0, min((rtt - delay_constraint) * task_priority, 10 * task_priority))
        # 延迟奖励（限制最大奖励）
        delay_reward = max(0, min((delay_constraint - rtt) * 0.5 * task_priority, 5 * task_priority))
        # 能耗惩罚（限制最大惩罚）
        energy_cost = min(energy * 0.1 / task_priority, 5 * task_priority)

        reward = base_reward + delay_reward - delay_penalty - energy_cost
        return max(reward, -5 * task_priority)


def train_ddpg(lr_actor=1e-6, lr_critic=5e-3, gamma=0.9, tau=0.01, episodes_num=EnvConfig.EPISODES):
    # 固定随机种子
    random.seed(EnvConfig.RANDOM_SEED)
    np.random.seed(EnvConfig.RANDOM_SEED)
    torch.manual_seed(EnvConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EnvConfig.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 初始化环境
    env = DDPGEnv(num_vehicles=EnvConfig.NUM_VEHICLES,
                  num_base_stations=EnvConfig.NUM_BASESTATIONS,
                  num_edge_servers=EnvConfig.NUM_EDGE_SERVERS)
    state_dim = env.num_vehicles * 4 + env.num_base_stations * 2 + env.num_edge_servers * 4 + 3
    action_dim = 3  # 对应LOCAL/EDGE/CLOUD三种动作

    # 初始化DDPG智能体
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        tau=tau
    )
    env.ddpg_agent = agent

    # 训练记录
    rewards_history = []
    actor_losses_history = []  # 存储所有step的actor loss
    critic_losses_history = []  # 存储所有step的critic loss

    logger.info(f"Start training DDPG with lr_actor={lr_actor}, lr_critic={lr_critic}, gamma={gamma}, tau={tau}")

    for episode in tqdm(range(episodes_num), desc="Training", unit="episode"):
        state = env.reset()
        total_reward = 0
        done = False
        episode_actor_losses = []
        episode_critic_losses = []

        while not done:
            # 选择动作
            action = agent.act(state)
            # 执行动作
            next_state, reward, done = env.step(action)

            # 将离散动作转换为连续向量（用于Critic网络）
            action_vec = np.zeros(action_dim)
            action_vec[action] = 1.0

            # 存储经验
            agent.remember(state, action_vec, reward, next_state, done)
            # 经验回放训练
            agent.replay()

            # 更新状态和奖励
            state = next_state
            total_reward += reward

            # 记录损失
            if agent.losses_actor:
                episode_actor_losses.append(agent.losses_actor[-1])
            if agent.losses_critic:
                episode_critic_losses.append(agent.losses_critic[-1])

        # 保存每轮总奖励
        rewards_history.append(total_reward)
        actor_losses_history.append(np.mean(episode_actor_losses))
        critic_losses_history.append(np.mean(episode_critic_losses))

    return rewards_history, env.metrics_episodes, (
    lr_actor, lr_critic, gamma, tau)
