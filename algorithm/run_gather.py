import json
import os
import time
from pathlib import Path

import torch
import random
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from algorithm.algo_config import EnvConfig
from algorithm.run_DDPG import train_ddpg
from algorithm.run_PPO import train_ppo
from algorithm.run_TD3 import train_td3
from algorithm.run_GCN_DDQN import train_gcn_ddqn
from algorithm.run_GCN_D3QN import train_gcn_d3qn
from algorithm.run_GAT_LSTM_DDQN import train_gat_lstm_ddqn
from algorithm.run_GAT_LSTM_D3QN import train_gat_lstm_d3qn
from algorithm.utils import smooth_data_ema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ALGORITHMS = {
    "DDPG": train_ddpg,
    "PPO": train_ppo,
    "TD3": train_td3,
    "GCN-DDQN": train_gcn_ddqn,
    "GCN-D3QN": train_gcn_d3qn,
    "GAT-LSTM-DDQN": train_gat_lstm_ddqn,
    "GLDVTO": train_gat_lstm_d3qn,
}

ALGORITHMS_List = []

if EnvConfig.IS_BASE:
    ALGORITHMS_List = ["DDPG", "PPO", "TD3", "GCN-DDQN", "GLDVTO"]
else:
    ALGORITHMS_List = ["GCN-DDQN", "GCN-D3QN", "GAT-LSTM-DDQN", "GLDVTO"]

def train_gather_algorithm(algorithm_name):
    train_func = ALGORITHMS[algorithm_name]
    rewards, metrics, _ = train_func()
    return rewards, metrics

def plot_algorithm_comparison(all_results, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    algos = list(all_results.keys())

    plt.figure(figsize=(10, 6))
    for i, algo in enumerate(algos):
        rewards = all_results[algo][0]
        color = colors[i % len(colors)]
        if algo == 'GLDVTO':
            color = '#9467bd'
        plt.plot(smooth_data_ema(rewards,alpha=0.9), label=algo, color=color, linewidth=1.5)
    plt.title("Reward Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Episode", fontsize=16)
    plt.ylabel("Total Reward", fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "reward_comparison.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    return

def plot_my_metric(all_results, save_dir):

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    algos = list(all_results.keys())

    plt.figure(figsize=(10, 6))
    key = 'average_transmission_delay'
    for i, algo in enumerate(algos):
        metrics_list = all_results[algo][1]
        values = []
        for metric in metrics_list:
            value = np.mean(metric[key])
            values.append(value)

        if len(values) > 0 and np.isnan(values[0]):
            values = values[1:]
        if len(values) > 0:
            smoothed = smooth_data_ema(values, alpha=0.9)
        else:
            smoothed = values
        if len(values) > 0:
            color = colors[i % len(colors)]
            if algo == 'GLDVTO':
                color = '#9467bd'
            plt.plot(smoothed, label=algo, color=color, linewidth=1.2, markersize=3)
    plt.title("Average Transmission Delay", fontsize=16)
    plt.ylabel("Delay (s)", fontsize=16)
    plt.xlabel("Episode", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig(save_dir / "average_transmission_delay.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    key = 'total_energy_consumption'
    for i, algo in enumerate(algos):
        metrics_list = all_results[algo][1]
        values = []
        for metric in metrics_list:
            value = np.mean(metric[key])
            values.append(value)
        if len(values) > 0 and np.isnan(values[0]):
            values = values[1:]
        if len(values) > 0:
            smoothed = smooth_data_ema(values, alpha=0.95)
        else:
            smoothed = values
        if len(values) > 0:
            color = colors[i % len(colors)]
            if algo == 'GLDVTO':
                color = '#9467bd'
            plt.plot(smoothed, label=algo, color=color, linewidth=1.2, markersize=3)
    plt.title("Total Energy Consumption", fontsize=16)
    plt.xlabel("Episode", fontsize=16)
    plt.ylabel("Energy", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig(save_dir / "total_energy_consumption.svg", format='svg', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    logger.info(f"Default Params: step ===> {EnvConfig.STEP}")
    logger.info(f"Default Params: episodes ===> {EnvConfig.EPISODES}")

    main_save_dir = "./algorithm_gather_comparison"
    os.makedirs(main_save_dir, exist_ok=True)

    random.seed(EnvConfig.RANDOM_SEED)
    np.random.seed(EnvConfig.RANDOM_SEED)
    torch.manual_seed(EnvConfig.RANDOM_SEED)
    torch.cuda.manual_seed_all(EnvConfig.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    all_results = {}

    start_time = time.time()

    for algorithm_name in ALGORITHMS_List:
        logger.info(f"Training : {algorithm_name}")
        if algorithm_name not in ALGORITHMS:
            logger.warning(f"{algorithm_name} not exist.")
            continue

        rewards, metrics_episodes = train_gather_algorithm(algorithm_name=algorithm_name)
        all_results[algorithm_name] = (rewards, metrics_episodes)
        logger.info(f"Finished: {algorithm_name}")
        print('\n\n')

    end_time = time.time()
    elapsed = end_time - start_time

    plot_algorithm_comparison(all_results, main_save_dir)
    plot_my_metric(all_results, main_save_dir)

    logger.info(f"All algorithm comparison experiments have been completed！Total time: {elapsed:.2f}s")
    logger.info(f"Spend {(end_time - start_time) // 60} min")
    logger.info(f"The result is saved in: {main_save_dir}")


if __name__ == "__main__":
    main()