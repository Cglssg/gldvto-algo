import copy
import logging
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
from algo_config import EnvConfig
from env import Simulator

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================== 网络定义 =====================
class DDQN(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=128, action_dim=3,
                 lstm_hidden_dim=64, lstm_layers=2, time_steps=5):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.time_steps = time_steps

        # GAT 层 - 空间特征提取
        self.conv1 = pyg_nn.GATConv(node_feature_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = pyg_nn.GATConv(hidden_dim * 4, hidden_dim, heads=2, concat=True)
        self.conv3 = pyg_nn.GATConv(hidden_dim * 2, hidden_dim // 2, heads=1, concat=False)

        # LSTM 层 - 时序特征提取
        self.lstm = nn.LSTM(
            input_size=8,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0
        )

        # DDQN 全连接层（移除Dueling结构）
        self.fc1 = nn.Linear(lstm_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, action_dim)

        # 激活与正则化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim // 2)

    def forward(self, data, time_series_features):
        # GAT 空间特征提取
        x, edge_index = data.x, data.edge_index
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv3(x, edge_index))
        x = self.layer_norm(x)

        # 图全局池化
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x_pool = pyg_nn.global_mean_pool(x, batch)

        # LSTM 时序特征提取
        ts = time_series_features.to(x.device)
        lstm_out, _ = self.lstm(ts)
        lstm_feat = lstm_out[:, -1, :]  # 取最后一个时间步的输出

        # DDQN 前向传播（无Dueling拆分）
        feat = self.relu(self.fc1(lstm_feat))
        feat = self.dropout(feat)
        feat = self.relu(self.fc2(feat))
        q_values = self.fc_out(feat)

        return q_values


# ===================== Agent =====================
class DDQNAgent:
    def __init__(self, node_feature_dim, edge_feature_dim, action_dim=3,
                 learning_rate=1e-5, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.99, time_steps=5):
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_min = epsilon_min  # 最小探索率
        self.epsilon_decay = epsilon_decay  # 探索率衰减
        self.time_steps = time_steps

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        # 双网络配置（DDQN核心）
        self.policy_net = DDQN(node_feature_dim, edge_feature_dim, action_dim=action_dim).to(self.device)
        self.target_net = DDQN(node_feature_dim, edge_feature_dim, action_dim=action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络仅用于评估，不训练

        # 优化器
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)

        # 经验回放池
        self.memory = deque(maxlen=20000)
        self.batch_size = 64

        # 训练参数
        self.train_step = 0
        self.target_update = 50  # 目标网络更新频率
        self.losses = deque(maxlen=10000)  # 损失记录

        # 时序特征缓冲
        self.ts_buffer = deque([torch.zeros(node_feature_dim) for _ in range(time_steps)], maxlen=time_steps)

    def get_ts_feat(self):
        """获取时序特征（time_steps长度的序列）"""
        return torch.stack(list(self.ts_buffer)).unsqueeze(0)

    def update_ts_buffer(self, graph_data):
        """更新时序特征缓冲"""
        with torch.no_grad():
            x = graph_data.x
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            pool = pyg_nn.global_mean_pool(x, batch)
            self.ts_buffer.append(pool.squeeze(0).cpu())

    def remember(self, g, ts, a, r, g2, ts2, done):
        """存储经验到回放池"""
        self.memory.append((g, ts, a, r, g2, ts2, done))

    def act(self, graph_data, evaluate=False):
        """选择动作（ε-贪心策略）"""
        # 评估模式下不探索
        if not evaluate and random.random() <= self.epsilon:
            return random.randrange(self.action_dim)

        # 贪心选择最优动作
        with torch.no_grad():
            self.update_ts_buffer(graph_data)
            ts = self.get_ts_feat().to(self.device)
            g = graph_data.to(self.device)
            q_values = self.policy_net(g, ts)
        return int(q_values.argmax().item())

    def replay(self):
        """经验回放（DDQN核心训练逻辑）"""
        if len(self.memory) < self.batch_size:
            return

        # 采样批次经验
        batch = random.sample(self.memory, self.batch_size)
        g1s, ts1s, acts, rews, g2s, ts2s, dones = zip(*batch)

        # 构建批次数据
        b1 = Batch.from_data_list([g.to("cpu") for g in g1s]).to(self.device)
        b2 = Batch.from_data_list([g.to("cpu") for g in g2s]).to(self.device)
        ts1 = torch.cat(ts1s).to(self.device)
        ts2 = torch.cat(ts2s).to(self.device)

        # 数据类型转换
        acts = torch.LongTensor(acts).to(self.device)
        rews = torch.FloatTensor(rews).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # 计算当前Q值（策略网络）
        q_current = self.policy_net(b1, ts1).gather(1, acts.unsqueeze(1)).squeeze()

        # DDQN核心：分离动作选择和Q值评估
        with torch.no_grad():
            # 1. 用策略网络选择下一状态的最优动作
            q_next_policy = self.policy_net(b2, ts2)
            best_actions = q_next_policy.argmax(1)
            # 2. 用目标网络评估该动作的Q值
            q_next_target = self.target_net(b2, ts2)
            q_next = q_next_target.gather(1, best_actions.unsqueeze(1)).squeeze()

        # 计算目标Q值
        q_target = rews + self.gamma * q_next * (~dones)

        # 损失计算与优化
        loss = F.mse_loss(q_current, q_target)
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # 梯度裁剪
        self.optimizer.step()

        # 探索率衰减
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_step += 1

        # 定期更新目标网络
        if self.train_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        """保存模型"""
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optim": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }, path)
        logger.info(f"模型已保存: {path}")

    def load_model(self, path):
        """加载模型"""
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optim"])
        self.epsilon = ckpt["epsilon"]
        logger.info(f"模型已加载: {path}")

    def reset(self):
        """重置Agent状态"""
        self.losses.clear()
        self.ts_buffer = deque([torch.zeros(self.node_feature_dim) for _ in range(self.time_steps)],
                               maxlen=self.time_steps)


# ===================== 环境 =====================
class GAT_LSTM_DDQNEnv:
    def __init__(self, agent=None):
        self.num_veh = EnvConfig.NUM_VEHICLES
        self.num_bs = EnvConfig.NUM_BASESTATIONS
        self.num_es = EnvConfig.NUM_EDGE_SERVERS
        self.agent = agent

        self.sim = Simulator(calculate_reward_fn=self.calc_reward)
        self.sim.setup_scenario(self.num_veh, self.num_bs, self.num_es)
        self.metrics = self.sim.metrics
        self.metrics_episodes = []

    def reset(self):
        """重置环境"""
        self.metrics_episodes.append(copy.deepcopy(self.metrics))
        self.sim = Simulator(calculate_reward_fn=self.calc_reward)
        self.sim.setup_scenario(self.num_veh, self.num_bs, self.num_es)
        self.metrics = self.sim.metrics
        if self.agent:
            self.agent.reset()
        return self.build_graph()

    def build_graph(self):
        """构建图数据（与原版本一致）"""
        nodes = []
        sim = self.sim

        # 车辆节点特征
        for v in sim.vehicles:
            cpu = v.cpu_usage / v.cpu_capacity if v.cpu_capacity > 0 else 0
            mem = v.memory_usage / v.memory_capacity if v.memory_capacity > 0 else 0
            px = v.position[0] / 12.0
            py = v.position[1] / 12.0
            task = len(v.tasks) / 10.0 if hasattr(v, "tasks") else 0
            bat = v.battery / 100.0 if hasattr(v, "battery") else 1.0
            nodes.append([px, py, cpu, mem, task, bat, 1.0, 1.0])

        # 基站节点特征
        for bs in sim.base_stations:
            bw = bs.bandwidth / 5000.0
            conn = len(bs.connected_vehicles) / self.num_veh
            px = bs.position[0] / 12.0
            py = bs.position[1] / 12.0
            nodes.append([px, py, bw, conn, 0, 0, 0, 0])

        # 边缘服务器节点特征
        for es in sim.edge_servers:
            cpu = es.cpu_usage / es.cpu_capacity if es.cpu_capacity > 0 else 0
            mem = es.memory_usage / es.memory_capacity if es.memory_capacity > 0 else 0
            load = es.current_load
            eng = es.energy_consumption / 1000.0
            px = es.position[0] / 12.0 if hasattr(es, "position") else 0
            py = es.position[1] / 12.0 if hasattr(es, "position") else 0
            nodes.append([px, py, cpu, mem, load, eng, 0, 0])

        # 云端节点特征
        cloud = sim.cloud_server
        cap = cloud.compute_capacity / 1000.0
        delay = cloud.communication_delay / 0.1
        eng = cloud.energy_consumption / 1000.0
        nodes.append([0.5, 0.5, cap, delay, eng, 0, 0, 0])

        # 边构建
        edge_idx = [[], []]
        edge_attr = []

        # 车-基站边
        vi = 0
        for v in sim.vehicles:
            bsi = self.num_veh
            for bs in sim.base_stations:
                d = v.calculate_distance(v.position, bs.position)
                if d < 2.0:
                    s = 1 - d / 2.0
                    edge_idx[0].append(vi)
                    edge_idx[1].append(bsi)
                    edge_attr.append([s])
                    edge_idx[0].append(bsi)
                    edge_idx[1].append(vi)
                    edge_attr.append([s])
                bsi += 1
            vi += 1

        # 基站-边缘服务器边
        bsi = self.num_veh
        for bs in sim.base_stations:
            esi = self.num_veh + self.num_bs
            for es in sim.edge_servers:
                if es in bs.connected_edge_servers:
                    edge_idx[0].append(bsi)
                    edge_idx[1].append(esi)
                    edge_attr.append([1.0])
                    edge_idx[0].append(esi)
                    edge_idx[1].append(bsi)
                    edge_attr.append([1.0])
            esi += 1
        bsi += 1

        # 转换为PyG数据格式
        x = torch.FloatTensor(nodes)
        ei = torch.LongTensor(edge_idx)
        ea = torch.FloatTensor(edge_attr)
        return Data(x=x, edge_index=ei, edge_attr=ea)

    def step(self, action):
        """执行动作，返回新状态"""
        r = self.sim.run_step(action)
        g = self.build_graph()
        if self.agent:
            self.agent.update_ts_buffer(g)
            ts = self.agent.get_ts_feat()
        else:
            ts = torch.zeros(1, 5, 8)
        done = self.sim.current_time >= EnvConfig.STEP
        return g, ts, r, done

    def calc_reward(self, success, rtt, delay_constraint, energy, task_priority=1):
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


# ===================== 训练函数 =====================
def train_gat_lstm_ddqn(lr=2e-5, gamma=0.95, epsilon=1.0, epsilon_decay=0.997, episodes_num=EnvConfig.EPISODES):
    # 固定随机种子
    random.seed(EnvConfig.RANDOM_SEED)
    np.random.seed(EnvConfig.RANDOM_SEED)
    torch.manual_seed(EnvConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EnvConfig.RANDOM_SEED)

    # 初始化Agent和环境
    agent = DDQNAgent(node_feature_dim=8, edge_feature_dim=1, epsilon=epsilon, action_dim=3,
                      learning_rate=lr, gamma=gamma, epsilon_decay=epsilon_decay)
    env = GAT_LSTM_DDQNEnv(agent=agent)

    # 训练记录
    rewards = []
    losses = []

    logger.info(f"开始训练 GAT-LSTM-DDQN | lr={lr} gamma={gamma} epsilon_decay={epsilon_decay}")

    # 训练主循环
    for ep in tqdm(range(episodes_num), desc="训练回合"):
        g = env.reset()
        total_r = 0
        done = False
        ep_loss = []

        while not done:
            ts = agent.get_ts_feat()
            a = agent.act(g)
            g2, ts2, r, done = env.step(a)
            agent.remember(g, ts, a, r, g2, ts2, done)
            agent.replay()
            g = g2
            total_r += r
            if agent.losses:
                ep_loss.append(agent.losses[-1])

        # 记录每回合奖励和损失
        rewards.append(total_r)
        losses.append(np.sum(ep_loss) if ep_loss else 0)

    return rewards, env.metrics_episodes, (lr, gamma, epsilon_decay)