import itertools
import copy
from algo_config import EnvConfig
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import matplotlib
from tqdm import tqdm
matplotlib.use('TkAgg')
from env import *
import os
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PPOTransition = namedtuple('PPOTransition',
                           ['state', 'action', 'log_prob', 'reward', 'next_state', 'done', 'value'])


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        logits = self.fc4(x)
        return F.softmax(logits, dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        value = self.fc4(x)
        return value


class PPOAgent:
    def __init__(self, state_dim, action_dim,
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.9,
                 clip_epsilon=0.2, K_epochs=10, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-5)

        self.memory = []
        self.memory_max_size = 10000
        self.warmup_steps = 1000

        self.actor_losses = deque(maxlen=10000)
        self.critic_losses = deque(maxlen=10000)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs = self.actor(state)
            value = self.critic(state)

        if evaluate:
            action = torch.argmax(action_probs, dim=1).item()
            log_prob = torch.log(action_probs[0, action]).item()
        else:
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()
            log_prob = action_dist.log_prob(torch.tensor(action, device=self.device)).item()

        return action, log_prob, value.item()

    def store_transition(self, transition):
        if len(self.memory) < self.memory_max_size:
            self.memory.append(transition)
        else:
            self.memory.pop(0)
            self.memory.append(transition)

    def compute_gae(self, rewards, dones, values, next_values, gamma=0.9, lam=0.95):
        advantages = []
        advantage = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantage = delta + gamma * lam * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)

        returns = np.array(advantages) + np.array(values)
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return advantages, returns

    def update(self):
        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return

        states = torch.FloatTensor([t.state for t in self.memory]).to(self.device)
        actions = torch.LongTensor([t.action for t in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([t.log_prob for t in self.memory]).to(self.device)
        rewards = [t.reward for t in self.memory]
        dones = [t.done for t in self.memory]
        values = [t.value for t in self.memory]

        next_states = torch.FloatTensor([t.next_state for t in self.memory]).to(self.device)
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze().cpu().numpy()

        advantages, returns = self.compute_gae(rewards, dones, values, next_values)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        for _ in range(self.K_epochs):
            indices = np.random.permutation(len(self.memory))
            for start in range(0, len(self.memory), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                action_probs = self.actor(batch_states)
                action_dist = torch.distributions.Categorical(action_probs)
                batch_log_probs = action_dist.log_prob(batch_actions)

                ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                current_values = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(current_values, batch_returns)

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(critic_loss.item())

        self.memory.clear()

    def save_model(self, filepath):
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
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
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
            self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
            self.actor_losses = deque(checkpoint.get('actor_losses', []), maxlen=10000)
            self.critic_losses = deque(checkpoint.get('critic_losses', []), maxlen=10000)
            logger.info(f"Model loaded from {filepath}")
        except FileNotFoundError:
            logger.error(f"Model file {filepath} not found")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")


class PPOEnv:
    def __init__(self, num_vehicles=EnvConfig.NUM_VEHICLES,
                 num_base_stations=EnvConfig.NUM_BASESTATIONS,
                 num_edge_servers=EnvConfig.NUM_EDGE_SERVERS):
        self.num_vehicles = num_vehicles
        self.num_base_stations = num_base_stations
        self.num_edge_servers = num_edge_servers

        self.current_state = None
        self.simulator = Simulator(calculate_reward_fn=self._calculate_reward)
        self.simulator.setup_scenario(num_vehicles, num_base_stations, num_edge_servers)
        self.metrics = self.simulator.metrics
        self.metrics_episodes = []

    def reset(self):
        self.metrics_episodes.append(copy.deepcopy(self.metrics))
        self.simulator = Simulator(calculate_reward_fn=self._calculate_reward)
        self.simulator.setup_scenario(self.num_vehicles, self.num_base_stations, self.num_edge_servers)
        self.metrics = self.simulator.metrics
        self.current_state = None
        state = self.normalize_state(self.get_state())
        return state

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

    def step(self, action):
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


def train_ppo(lr_actor=5e-5, lr_critic=1e-5, gamma=0.95,
              clip_epsilon=0.2, episodes_num=EnvConfig.EPISODES):
    random.seed(EnvConfig.RANDOM_SEED)
    np.random.seed(EnvConfig.RANDOM_SEED)
    torch.manual_seed(EnvConfig.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EnvConfig.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env = PPOEnv(num_vehicles=EnvConfig.NUM_VEHICLES,
                 num_base_stations=EnvConfig.NUM_BASESTATIONS,
                 num_edge_servers=EnvConfig.NUM_EDGE_SERVERS)
    state_dim = env.num_vehicles * 4 + env.num_base_stations * 2 + env.num_edge_servers * 4 + 3
    action_dim = 3

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        clip_epsilon=clip_epsilon
    )

    rewards_history = []
    actor_loss_history = []
    critic_loss_history = []
    logger.info(f"Start training PPO with lr_actor={lr_actor}, lr_critic={lr_critic}, gamma={gamma}")

    for episode in tqdm(range(episodes_num), desc="Training", unit="episode"):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done = env.step(action)
            transition = PPOTransition(
                state=state,
                action=action,
                log_prob=log_prob,
                reward=reward,
                next_state=next_state,
                done=done,
                value=value
            )
            agent.store_transition(transition)

            state = next_state
            total_reward += reward

        agent.update()

        rewards_history.append(total_reward)
        if agent.actor_losses:
            actor_loss_history.append(np.mean(list(agent.actor_losses)[-100:]))
        if agent.critic_losses:
            critic_loss_history.append(np.mean(list(agent.critic_losses)[-100:]))

    return rewards_history, env.metrics_episodes, (lr_actor, lr_critic, gamma)