import json
import os
import pickle
import time
import argparse
import torch
import random
from datetime import datetime
from pathlib import Path
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
from algorithm.utils import smooth_data_ema, convert_to_json_serializable
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

def plot_algorithm_comparison_old(all_results, save_dir):
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

def plot_algorithm_comparison(all_results, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    algos = list(all_results.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, algo in enumerate(algos):
        rewards = all_results[algo][0]
        if len(rewards) > 0:
            smoothed = smooth_data_ema(rewards, alpha=0.9)
        else:
            smoothed = rewards
        if len(smoothed) > 0:
            color = colors[i % len(colors)]
            if algo == 'GLDVTO':
                color = '#9b8ac9'
            ax.plot(
                smoothed,
                label=algo,
                color=color,
                linewidth=1.2
            )
    ax.set_xlabel("Episode", fontsize=17)
    ax.set_ylabel("Total Reward", fontsize=17)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(alpha=0.25, linestyle='--', linewidth=0.6)
    legend = ax.legend(
        loc='lower left',
        bbox_to_anchor=(0.0, 1.01, 1.0, 0.12),
        mode='expand',
        ncol=len(algos),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor='gray',
        fontsize=14,
        handlelength=2.0,
        handletextpad=0.6,
        columnspacing=1.2,
        borderpad=0.45,
        borderaxespad=0.0
    )
    legend.get_frame().set_linewidth(0.8)
    fig.subplots_adjust(top=0.85, left=0.11, right=0.98, bottom=0.12)
    fig.savefig(
        save_dir / "reward_comparison.svg",
        format='svg',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

def plot_my_metric(all_results, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    algos = list(all_results.keys())
    def draw_metric(key, ylabel, save_name, alpha=0.9):
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, algo in enumerate(algos):
            metrics_list = all_results[algo][1]
            values = []
            for metric in metrics_list:
                value = np.mean(metric[key])
                values.append(value)
            if len(values) > 0 and np.isnan(values[0]):
                values = values[1:]
            if len(values) > 0:
                smoothed = smooth_data_ema(values, alpha=alpha)
            else:
                smoothed = values
            if len(smoothed) > 0:
                color = colors[i % len(colors)]
                if algo == 'GLDVTO':
                    color = '#9b8ac9'
                ax.plot(
                    smoothed,
                    label=algo,
                    color=color,
                    linewidth=1.2
                )
        ax.set_xlabel("Episode", fontsize=17)
        ax.set_ylabel(ylabel, fontsize=17)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(alpha=0.25, linestyle='--', linewidth=0.6)
        legend = ax.legend(
            loc='lower left',
            bbox_to_anchor=(0.0, 1.01, 1.0, 0.12),
            mode='expand',
            ncol=len(algos),
            frameon=True,
            fancybox=False,
            framealpha=1.0,
            edgecolor='gray',
            fontsize=14,
            handlelength=2.0,
            handletextpad=0.6,
            columnspacing=1.2,
            borderpad=0.45,
            borderaxespad=0.0
        )
        legend.get_frame().set_linewidth(0.8)
        fig.subplots_adjust(top=0.85, left=0.11, right=0.98, bottom=0.12)
        fig.savefig(save_dir / save_name, format='svg', dpi=300, bbox_inches='tight')
        plt.close(fig)
    draw_metric(
        key='average_transmission_delay',
        ylabel='Total Task Latency',
        save_name='total_task_latency.svg',
        alpha=0.9
    )
    draw_metric(
        key='total_energy_consumption',
        ylabel='Total Task Energy',
        save_name='total_task_energy.svg',
        alpha=0.95
    )

def parse_args():
    parser = argparse.ArgumentParser(description="GLDVTO algorithm comparison training and plotting")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Read saved training results directly for plotting without retraining"
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="File path of training result cache; use default path if not specified"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./algorithm_gather_comparison",
        help="Directory for saving images and cache files"
    )
    parser.add_argument(
        "--no-save-cache",
        action="store_true",
        help="Do not save training result cache after training"
    )
    return parser.parse_args()

def build_default_cache_path(save_dir):
    exp_name = "base" if EnvConfig.IS_BASE else "ablation"
    cache_dir = Path(save_dir) / "cached_results"
    cache_name = f"{exp_name}_ep{EnvConfig.EPISODES}_step{EnvConfig.STEP}_seed{EnvConfig.RANDOM_SEED}.pkl"
    return cache_dir / cache_name

def save_training_results(all_results, cache_path):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "saved_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_base": EnvConfig.IS_BASE,
        "episodes": EnvConfig.EPISODES,
        "step": EnvConfig.STEP,
        "random_seed": EnvConfig.RANDOM_SEED,
        "algorithms": list(all_results.keys()),
    }
    payload = {
        "metadata": metadata,
        "all_results": all_results,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    summary_path = cache_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(convert_to_json_serializable(metadata), f, ensure_ascii=False, indent=2)
    logger.info(f"Training result cache saved: {cache_path}")
    logger.info(f"Cache summary saved: {summary_path}")

def load_training_results(cache_path):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Training result cache file not found: {cache_path}")
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "all_results" in payload:
        all_results = payload["all_results"]
        metadata = payload.get("metadata", {})
    else:
        all_results = payload
        metadata = {}
    if not isinstance(all_results, dict) or len(all_results) == 0:
        raise ValueError(f"Abnormal cache file content, cannot plot: {cache_path}")
    logger.info(f"Loaded training result cache: {cache_path}")
    if metadata:
        logger.info(
            "Cache Info: "
            f"episodes={metadata.get('episodes')}, "
            f"step={metadata.get('step')}, "
            f"seed={metadata.get('random_seed')}, "
            f"algorithms={metadata.get('algorithms')}"
        )
    return all_results, metadata

def main():
    args = parse_args()

    logger.info(f"Default Params: step ===> {EnvConfig.STEP}")
    logger.info(f"Default Params: episodes ===> {EnvConfig.EPISODES}")
    main_save_dir = args.save_dir
    os.makedirs(main_save_dir, exist_ok=True)
    cache_path = Path(args.cache_path) if args.cache_path else build_default_cache_path(main_save_dir)
    logger.info(f"Cache path: {cache_path}")
    start_time = time.time()
    if args.use_cache:
        all_results, _ = load_training_results(cache_path)
    else:
        random.seed(EnvConfig.RANDOM_SEED)
        np.random.seed(EnvConfig.RANDOM_SEED)
        torch.manual_seed(EnvConfig.RANDOM_SEED)
        torch.cuda.manual_seed_all(EnvConfig.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        all_results = {}
        for algorithm_name in ALGORITHMS_List:
            logger.info(f"Training : {algorithm_name}")
            if algorithm_name not in ALGORITHMS:
                logger.warning(f"{algorithm_name} not exist.")
                continue
            rewards, metrics_episodes = train_gather_algorithm(algorithm_name=algorithm_name)
            all_results[algorithm_name] = (rewards, metrics_episodes)
            logger.info(f"Finished: {algorithm_name}")
            print('\n\n')
        if not args.no_save_cache:
            save_training_results(all_results, cache_path)
    elapsed = time.time() - start_time
    plot_algorithm_comparison(all_results, main_save_dir)
    plot_my_metric(all_results, main_save_dir)
    logger.info(f"All algorithm comparison experiments have been completed！Total time: {elapsed:.2f}s")
    logger.info(f"spend {elapsed // 60} min")
    logger.info(f"The result is saved in: {main_save_dir}")
if __name__ == "__main__":
    main()