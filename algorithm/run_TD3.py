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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义根目录和子目录
ROOT_RESULT_DIR = "result_td3"
MODEL_DIR = os.path.join(ROOT_RESULT_DIR, "model")
PLOT_DIR = os.path.join(ROOT_RESULT_DIR, "plt")
METRICS_PLOT_DIR = os.path.join(PLOT_DIR, "metrics_9grid")

# 确保目录存在
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRICS_PLOT_DIR, exist_ok=True)


# ========== TD3 核心网络定义 ==========
class Actor(nn.Module):
    """TD3 Actor网络（确定性策略）"""

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
        # 输出缩放到[-action_bound, action_bound]
        action = torch.tanh(self.fc4(x)) * self.action_bound
        return action


class Critic(nn.Module):
    """TD3 Critic网络（双Q网络）"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1网络
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc4 = nn.Linear(hidden_dim, 1)

        # Q2网络
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        # Q1前向传播
        q1 = torch.relu(self.q1_fc1(sa))
        q1 = torch.relu(self.q1_fc2(q1))
        q1 = torch.relu(self.q1_fc3(q1))
        q1 = self.q1_fc4(q1)

        # Q2前向传播
        q2 = torch.relu(self.q2_fc1(sa))
        q2 = torch.relu(self.q2_fc2(q2))
        q2 = torch.relu(self.q2_fc3(q2))
        q2 = self.q2_fc4(q2)

        return q1, q2

    def q1_forward(self, state, action):
        """仅Q1前向传播（用于策略更新）"""
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.q1_fc1(sa))
        q1 = torch.relu(self.q1_fc2(q1))
        q1 = torch.relu(self.q1_fc3(q1))
        q1 = self.q1_fc4(q1)
        return q1


# ========== TD3智能体 ==========
class TD3Agent:
    """TD3智能体"""

    def __init__(self, state_dim, action_dim,
                 learning_rate=3e-4, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau  # 目标网络软更新系数
        self.policy_noise = policy_noise  # 策略噪声
        self.noise_clip = noise_clip  # 噪声裁剪
        self.policy_freq = policy_freq  # 策略更新频率（延迟更新）

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化网络
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        # 初始化目标网络参数
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate, weight_decay=1e-5)

        # 经验回放缓冲区
        self.memory = deque(maxlen=100000)
        self.batch_size = 256
        self.warmup_steps = 1000  # 预热步数
        self.train_step = 0

        # 噪声参数（探索用）
        self.exploration_noise = 0.1
        self.action_bound = 1.0

        # 记录损失
        self.actor_losses = deque(maxlen=10000)
        self.critic_losses = deque(maxlen=10000)

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, evaluate=False):
        """选择动作（评估模式禁用探索噪声）"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]

        # 评估模式不加噪声
        if not evaluate:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = (action + noise).clip(-self.action_bound, self.action_bound)

        # 转换为离散动作（适配原环境的3个动作）
        action_idx = np.argmax(action) if len(action) > 0 else 0
        return action_idx

    def replay(self):
        """经验回放训练"""
        # 预热机制
        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return

        # 采样批次数据
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device).unsqueeze(1)

        # ========== 训练Critic网络 ==========
        with torch.no_grad():
            # 目标动作（添加噪声）
            next_actions = self.actor_target(next_states)
            noise = torch.clamp(
                torch.normal(0, self.policy_noise, size=next_actions.shape).to(self.device),
                -self.noise_clip, self.noise_clip
            )
            next_actions = torch.clamp(next_actions + noise, -self.action_bound, self.action_bound)

            # 计算目标Q值（取双Q最小值）
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones.float()) * self.gamma * target_q

        # 当前Q值
        current_q1, current_q2 = self.critic(states, actions)

        # 计算Critic损失（Huber Loss）
        critic_loss = F.smooth_l1_loss(current_q1, target_q) + F.smooth_l1_loss(current_q2, target_q)
        self.critic_losses.append(critic_loss.item())

        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # ========== 延迟更新Actor网络 ==========
        if self.train_step % self.policy_freq == 0:
            # 计算Actor损失（最大化Q1值）
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
            self.actor_losses.append(actor_loss.item())

            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            # 软更新目标网络
            self._soft_update(self.critic_target, self.critic, self.tau)
            self._soft_update(self.actor_target, self.actor, self.tau)

        self.train_step += 1

    def _soft_update(self, target, source, tau):
        """软更新目标网络参数"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save_model(self, filepath):
        """保存模型"""
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
        """加载模型"""
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
        """重置损失记录"""
        self.actor_losses.clear()
        self.critic_losses.clear()

class BaseEnv:
    """车辆边缘计算环境（仅使用DQN）"""
    def __init__(self, num_vehicles=EnvConfig.NUM_VEHICLES,
                 num_base_stations=EnvConfig.NUM_BASESTATIONS,
                 num_edge_servers=EnvConfig.NUM_EDGE_SERVERS):
        self.num_vehicles = num_vehicles
        self.num_base_stations = num_base_stations
        self.num_edge_servers = num_edge_servers

        self.current_state = None  # 缓存当前状态，供回调函数使用

        self.simulator = Simulator(calculate_reward_fn = self._calculate_reward)

        self.simulator.setup_scenario(num_vehicles, num_base_stations, num_edge_servers)
        # 一轮的metrics(直接复用simulator的metrics，保证数据统一)
        self.metrics = self.simulator.metrics
        # 每一轮的metrics
        self.metrics_episodes = []

    def reset(self):
        """重置环境"""
        # 保存上一轮的metrics
        # 为 metric 中的 list 做消除异常值 0 ，平滑处理
        # keys = list(self.metrics.keys())
        # for key in keys:
        #     if type(self.metrics[key]) == 'list':
        #         self.metrics[key] = self.metrics[key]
        self.metrics_episodes.append(copy.deepcopy(self.metrics))

        self.simulator = Simulator(calculate_reward_fn = self._calculate_reward)
        self.simulator.setup_scenario(self.num_vehicles, self.num_base_stations, self.num_edge_servers)
        self.metrics = self.simulator.metrics  # 重置后同步metrics
        self.current_state = None


    # def normalize_state(self, state):
    #     """状态归一化（避免量纲影响）"""
    #     state = np.array(state, dtype=np.float32)
    #     max_vals = np.max(np.abs(state))
    #     if max_vals > 0:
    #         state = state / max_vals
    #     return state

    # run_DQN.py - DQNEnv 类修改
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
        """修复特征引用错误"""
        state_features = []
        # 车辆特征（4维：x, y, cpu利用率, 内存利用率）
        for vehicle in self.simulator.vehicles:
            cpu_util = vehicle.cpu_usage / vehicle.cpu_capacity if hasattr(vehicle,
                                                                           'cpu_usage') and vehicle.cpu_capacity > 0 else 0.0
            mem_util = vehicle.memory_usage / vehicle.memory_capacity if hasattr(vehicle,
                                                                                 'memory_usage') and vehicle.memory_capacity > 0 else 0.0
            state_features.extend([
                vehicle.position[0] / 12.0,  # 地图范围0-12，提前归一化
                vehicle.position[1] / 12.0,
                np.clip(cpu_util, 0, 1),
                np.clip(mem_util, 0, 1)
            ])

        # 基站特征（2维：归一化带宽, 归一化连接车辆数）
        for bs in self.simulator.base_stations:
            bandwidth_norm = bs.bandwidth / 5000.0  # 基站带宽最大5000
            connected_norm = len(bs.connected_vehicles) / self.num_vehicles if self.num_vehicles > 0 else 0.0
            state_features.extend([
                np.clip(bandwidth_norm, 0, 1),
                np.clip(connected_norm, 0, 1)
            ])

        # 修复：边缘服务器特征引用自身属性
        for es in self.simulator.edge_servers:
            cpu_util = es.cpu_usage / es.cpu_capacity if hasattr(es, 'cpu_usage') and es.cpu_capacity > 0 else 0.0
            mem_util = es.memory_usage / es.memory_capacity if hasattr(es,
                                                                       'memory_usage') and es.memory_capacity > 0 else 0.0
            state_features.extend([
                np.clip(cpu_util, 0, 1),
                np.clip(mem_util, 0, 1),
                np.clip(es.current_load, 0, 1),
                es.energy_consumption / 1000.0  # 边缘能耗最大1000
            ])

        # 云端特征（3维）
        cloud = self.simulator.cloud_server
        state_features.extend([
            cloud.compute_capacity / 1000.0,
            cloud.communication_delay / 0.1,  # 云端延迟最大0.1s
            cloud.energy_consumption / 1000.0
        ])

        # 固定维度
        target_length = self.num_vehicles * 4 + self.num_base_stations * 2 + self.num_edge_servers * 4 + 3
        if len(state_features) < target_length:
            state_features += [0.0] * (target_length - len(state_features))
        else:
            state_features = state_features[:target_length]
        return np.array(state_features, dtype=np.float32)

    def step(self,action):
        """执行动作并返回新的状态、奖励和完成标志"""

        # 【核心】直接调用 simulator.run_step()，它会内部调用我们注入的 _dqn_offload_decision_callback
        reward = self.simulator.run_step(action)

        # 获取下一个状态
        next_state = self.normalize_state(self.get_state())
        self.current_state = next_state

        # 判断结束
        done = self.simulator.current_time >= EnvConfig.STEP

        return next_state, reward, done

    # def _calculate_reward(self, success, rtt, delay_constraint, energy, task_priority=1):
    #     task_priority = max(1, min(5, task_priority))
    #     if not success:
    #         return -30 * task_priority  # 失败惩罚加大到3倍基础奖励
    #     # 基础奖励
    #     base_reward = 30 * task_priority
    #     # 延迟惩罚（超约束时惩罚翻倍，系数5.0）
    #     delay_penalty = max(0, (rtt - delay_constraint) * 5.0 * task_priority)
    #     # 延迟奖励（提前完成奖励，系数3.0）
    #     delay_reward = max(0, (delay_constraint - rtt) * 3.0 * task_priority)
    #     # 能耗惩罚（系数2.0，按优先级调整，高优先级对能耗更敏感）
    #     energy_cost = min(energy * 2.0 / task_priority, 15 * task_priority)
    #     # 限制奖励范围，避免梯度爆炸
    #     reward = base_reward + delay_reward - delay_penalty - energy_cost
    #     return np.clip(reward, -50, 50)
    def _calculate_reward(self, success, rtt, delay_constraint, energy, task_priority=1):
        """优化奖励函数（增加约束）"""
        task_priority = max(1, min(5, task_priority))  # 限制优先级范围
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
        return max(reward, -5 * task_priority)  # 限制最小奖励

class TD3Env(BaseEnv):
    """TD3环境（复用DQNEnv逻辑，适配连续动作）"""

    def step(self, action_idx):
        """执行动作（转换离散动作索引为连续动作）"""
        # 将离散动作索引转换为连续动作向量
        action = np.zeros(self.num_edge_servers + 1)  # 适配动作维度
        action[action_idx] = 1.0

        # 执行原环境逻辑
        reward = self.simulator.run_step(action_idx)
        next_state = self.normalize_state(self.get_state())
        self.current_state = next_state
        done = self.simulator.current_time >= EnvConfig.STEP

        # 返回连续动作格式的经验
        return next_state, reward, done


# ========== 训练函数 ==========
def train_td3(lr=1e-5, gamma=0.8, tau=0.005, policy_noise=0.5, episodes_num=EnvConfig.EPISODES):
    # 固定随机种子
    random.seed(EnvConfig.RANDOM_SEED)
    np.random.seed(EnvConfig.RANDOM_SEED)
    torch.manual_seed(EnvConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EnvConfig.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 初始化环境
    env = TD3Env(
        num_vehicles=EnvConfig.NUM_VEHICLES,
        num_base_stations=EnvConfig.NUM_BASESTATIONS,
        num_edge_servers=EnvConfig.NUM_EDGE_SERVERS
    )

    # 状态/动作维度
    state_dim = env.num_vehicles * 4 + env.num_base_stations * 2 + env.num_edge_servers * 4 + 3
    action_dim = 3  # 保持原动作维度

    # 初始化TD3智能体
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=lr,
        gamma=gamma,
        tau=tau,
        policy_noise=policy_noise
    )

    # 训练记录
    rewards_history = []
    actor_losses_history = []
    critic_losses_history = []
    logger.info(f"Start training TD3 with lr={lr}, gamma={gamma}, tau={tau}, policy_noise={policy_noise}")

    # 训练主循环
    for episode in tqdm(range(episodes_num), desc="Training TD3", unit="episode"):
        state = env.normalize_state(env.get_state())
        total_reward = 0
        done = False
        episode_actor_losses = []
        episode_critic_losses = []

        while not done:
            # 选择动作
            action_idx = agent.act(state)
            # 执行动作
            next_state, reward, done = env.step(action_idx)
            # 转换为连续动作（用于存储经验）
            action = np.zeros(action_dim)
            action[action_idx] = 1.0
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            # 经验回放
            agent.replay()

            # 更新状态和奖励
            state = next_state
            total_reward += reward

            # 记录损失
            if agent.actor_losses:
                episode_actor_losses.append(agent.actor_losses[-1])
            if agent.critic_losses:
                episode_critic_losses.append(agent.critic_losses[-1])

        # 保存每轮结果
        rewards_history.append(total_reward)
        actor_losses_history.append(np.mean(episode_actor_losses) if episode_actor_losses else 0)
        critic_losses_history.append(np.mean(episode_critic_losses) if episode_critic_losses else 0)

        # 重置环境
        env.reset()

    # 保存模型
    model_path = os.path.join(MODEL_DIR, f"td3_lr{lr}_gamma{gamma}_tau{tau}_noise{policy_noise}.pth")
    agent.save_model(model_path)

    return rewards_history, env.metrics_episodes, (lr, gamma, tau, policy_noise)