import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_training_cache(pkl_path, summary_path=None):
    pkl_path = Path(pkl_path)

    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "all_results" in payload:
        all_results = payload["all_results"]
        metadata = payload.get("metadata", {})
    else:
        all_results = payload
        metadata = {}

    if not isinstance(all_results, dict) or len(all_results) == 0:
        raise ValueError("Invalid pkl structure: all_results is empty or not a dict.")

    summary = None
    if summary_path is None:
        default_summary = pkl_path.with_suffix(".summary.json")
        if default_summary.exists():
            summary_path = default_summary

    if summary_path is not None and Path(summary_path).exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    return all_results, metadata, summary


def safe_mean(values):
    if values is None:
        return np.nan
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if len(arr) > 0 else np.nan


def smooth_ema(values, alpha=0.9):

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    smoothed = np.zeros_like(values, dtype=float)

    finite_idx = np.where(np.isfinite(values))[0]
    if len(finite_idx) == 0:
        return values

    first = finite_idx[0]
    smoothed[:first] = np.nan
    smoothed[first] = values[first]

    for i in range(first + 1, len(values)):
        if np.isfinite(values[i]):
            smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * values[i]
        else:
            smoothed[i] = smoothed[i - 1]

    return smoothed


def extract_episode_metric(metrics_episodes, metric_key):

    curve = []

    for ep_metric in metrics_episodes:
        if not isinstance(ep_metric, dict) or metric_key not in ep_metric:
            curve.append(np.nan)
        else:
            curve.append(safe_mean(ep_metric[metric_key]))

    curve = np.asarray(curve, dtype=float)

    if len(curve) > 0 and np.isnan(curve[0]):
        curve = curve[1:]

    return curve


def extract_all_curves(all_results):
    curves = {}

    for algo, result in all_results.items():
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            raise ValueError(f"Invalid result format for {algo}. Expected (rewards, metrics_episodes).")

        rewards = np.asarray(result[0], dtype=float)
        metrics_episodes = result[1]

        latency = extract_episode_metric(metrics_episodes, "average_transmission_delay")
        energy = extract_episode_metric(metrics_episodes, "total_energy_consumption")

        curves[algo] = {
            "reward": rewards,
            "latency": latency,
            "energy": energy,
        }

    return curves


def smooth_all_curves(raw_curves, reward_alpha=0.9, latency_alpha=0.9, energy_alpha=0.95):
    smoothed = {}

    for algo, data in raw_curves.items():
        smoothed[algo] = {
            "reward": smooth_ema(data["reward"], alpha=reward_alpha),
            "latency": smooth_ema(data["latency"], alpha=latency_alpha),
            "energy": smooth_ema(data["energy"], alpha=energy_alpha),
        }

    return smoothed


def get_overall_mean(curve):
    return safe_mean(curve)


def get_tail_mean(curve, tail_ratio=0.25):
    curve = np.asarray(curve, dtype=float)
    curve = curve[np.isfinite(curve)]

    if len(curve) == 0:
        return np.nan

    start = int(len(curve) * (1 - tail_ratio))
    return safe_mean(curve[start:])


def calc_reduction(baseline, proposed):

    if not np.isfinite(baseline) or not np.isfinite(proposed) or baseline == 0:
        return np.nan
    return (baseline - proposed) / baseline * 100


def calc_improvement(proposed, baseline):

    if not np.isfinite(baseline) or not np.isfinite(proposed) or baseline == 0:
        return np.nan
    return (proposed - baseline) / abs(baseline) * 100


def convergence_episode(curve, mode="higher", tail_ratio=0.25, tol=0.05, stable_window=10):
    curve = np.asarray(curve, dtype=float)

    if np.isfinite(curve).sum() < stable_window:
        return np.nan

    stable_value = get_tail_mean(curve, tail_ratio=tail_ratio)

    if not np.isfinite(stable_value):
        return np.nan

    if mode == "higher":
        threshold = stable_value * (1 - tol)
        good = curve >= threshold
    else:
        threshold = stable_value * (1 + tol)
        good = curve <= threshold

    for i in range(0, len(curve) - stable_window + 1):
        values = curve[i:i + stable_window]
        if np.all(np.isfinite(values)) and np.all(good[i:i + stable_window]):
            return i + 1

    return np.nan


def calc_convergence_speedup(baseline_ep, proposed_ep):
    if not np.isfinite(baseline_ep) or not np.isfinite(proposed_ep) or baseline_ep == 0:
        return np.nan
    return (baseline_ep - proposed_ep) / baseline_ep * 100


def analyze_curves(
    curves,
    proposed_name="GLDVTO",
    tail_ratio=0.25,
    tol=0.05,
    stable_window=10,
):

    if proposed_name not in curves:
        raise ValueError(f"{proposed_name} not found. Available: {list(curves.keys())}")

    rows = []

    for algo, data in curves.items():
        reward = data["reward"]
        latency = data["latency"]
        energy = data["energy"]

        rows.append({
            "Algorithm": algo,

            "Reward Overall Mean": get_overall_mean(reward),
            "Reward Stable Mean": get_tail_mean(reward, tail_ratio),
            "Reward Conv. Ep.": convergence_episode(
                reward, mode="higher", tail_ratio=tail_ratio,
                tol=tol, stable_window=stable_window
            ),

            "Latency Overall Mean": get_overall_mean(latency),
            "Latency Stable Mean": get_tail_mean(latency, tail_ratio),
            "Latency Conv. Ep.": convergence_episode(
                latency, mode="lower", tail_ratio=tail_ratio,
                tol=tol, stable_window=stable_window
            ),

            "Energy Overall Mean": get_overall_mean(energy),
            "Energy Stable Mean": get_tail_mean(energy, tail_ratio),
            "Energy Conv. Ep.": convergence_episode(
                energy, mode="lower", tail_ratio=tail_ratio,
                tol=tol, stable_window=stable_window
            ),
        })

    metrics_df = pd.DataFrame(rows)
    proposed = metrics_df[metrics_df["Algorithm"] == proposed_name].iloc[0]

    imp_rows = []

    for _, row in metrics_df.iterrows():
        algo = row["Algorithm"]

        if algo == proposed_name:
            continue

        imp_rows.append({
            "Compared Algorithm": algo,

            "Latency Reduction Overall (%)": calc_reduction(
                row["Latency Overall Mean"], proposed["Latency Overall Mean"]
            ),
            "Latency Reduction Stable (%)": calc_reduction(
                row["Latency Stable Mean"], proposed["Latency Stable Mean"]
            ),

            "Energy Reduction Overall (%)": calc_reduction(
                row["Energy Overall Mean"], proposed["Energy Overall Mean"]
            ),
            "Energy Reduction Stable (%)": calc_reduction(
                row["Energy Stable Mean"], proposed["Energy Stable Mean"]
            ),

            "Reward Improvement Overall (%)": calc_improvement(
                proposed["Reward Overall Mean"], row["Reward Overall Mean"]
            ),
            "Reward Improvement Stable (%)": calc_improvement(
                proposed["Reward Stable Mean"], row["Reward Stable Mean"]
            ),

            "Reward Conv. Speedup (%)": calc_convergence_speedup(
                row["Reward Conv. Ep."], proposed["Reward Conv. Ep."]
            ),
            "Latency Conv. Speedup (%)": calc_convergence_speedup(
                row["Latency Conv. Ep."], proposed["Latency Conv. Ep."]
            ),
            "Energy Conv. Speedup (%)": calc_convergence_speedup(
                row["Energy Conv. Ep."], proposed["Energy Conv. Ep."]
            ),
        })

    improvement_df = pd.DataFrame(imp_rows)

    return metrics_df, improvement_df

def plot_curves(raw_curves, smoothed_curves, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for key, ylabel in [
        ("reward", "Total Reward"),
        ("latency", "Total Task Latency"),
        ("energy", "Total Task Energy"),
    ]:
        plot_one_metric(
            curves=smoothed_curves,
            key=key,
            ylabel=ylabel,
            save_path=save_dir / f"{key}_curve.png",
            title=f"{ylabel} Smoothed"
        )

        # plot_raw_and_smoothed(
        #     raw_curves=raw_curves,
        #     smoothed_curves=smoothed_curves,
        #     key=key,
        #     ylabel=ylabel,
        #     save_path=save_dir / f"{key}_raw_vs_smoothed.png"
        # )


def plot_one_metric(curves, key, ylabel, save_path, title=None):
    plt.figure(figsize=(10, 6))

    for algo, data in curves.items():
        y = np.asarray(data[key], dtype=float)
        plt.plot(y, label=algo, linewidth=1.5)

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_raw_and_smoothed(raw_curves, smoothed_curves, key, ylabel, save_path):
    plt.figure(figsize=(10, 6))

    for algo in raw_curves.keys():
        raw = np.asarray(raw_curves[algo][key], dtype=float)
        smoothed = np.asarray(smoothed_curves[algo][key], dtype=float)

        plt.plot(raw, alpha=0.25, linewidth=0.8)
        plt.plot(smoothed, label=f"{algo}", linewidth=1.5)

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel}: Raw and Smoothed")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def analyze_training_result_file(
    pkl_path,
    summary_path=None,
    proposed_name="GLDVTO",
    output_dir="./analysis_output",
    tail_ratio=0.25,
    tol=0.05,
    stable_window=10,
    reward_alpha=0.9,
    latency_alpha=0.9,
    energy_alpha=0.95,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results, metadata, summary = load_training_cache(pkl_path, summary_path)

    print("========== Metadata from PKL ==========")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    if summary is not None:
        print("\n========== Summary JSON ==========")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    raw_curves = extract_all_curves(all_results)

    smoothed_curves = smooth_all_curves(
        raw_curves,
        reward_alpha=reward_alpha,
        latency_alpha=latency_alpha,
        energy_alpha=energy_alpha,
    )

    metrics_df, improvement_df = analyze_curves(
        curves=smoothed_curves,
        proposed_name=proposed_name,
        tail_ratio=tail_ratio,
        tol=tol,
        stable_window=stable_window,
    )

    metrics_csv = output_dir / "metrics_summary_smoothed.csv"
    improvement_csv = output_dir / "improvement_summary_smoothed.csv"

    metrics_df.to_csv(metrics_csv, index=False, encoding="utf-8-sig")
    improvement_df.to_csv(improvement_csv, index=False, encoding="utf-8-sig")

    metrics_tex = output_dir / "metrics_summary_smoothed.data.tex"
    improvement_tex = output_dir / "improvement_summary_smoothed.data.tex"

    with open(metrics_tex, "w", encoding="utf-8") as f:
        f.write(metrics_df.round(4).to_latex(index=False, escape=False))

    with open(improvement_tex, "w", encoding="utf-8") as f:
        f.write(improvement_df.round(2).to_latex(index=False, escape=False))

    np.savez(
        output_dir / "smoothed_curves.npz",
        **{
            f"{algo}_{metric}": np.asarray(values[metric], dtype=float)
            for algo, values in smoothed_curves.items()
            for metric in ["reward", "latency", "energy"]
        }
    )

    np.savez(
        output_dir / "raw_curves.npz",
        **{
            f"{algo}_{metric}": np.asarray(values[metric], dtype=float)
            for algo, values in raw_curves.items()
            for metric in ["reward", "latency", "energy"]
        }
    )

    plot_curves(raw_curves, smoothed_curves, output_dir)

    print("\n========== Metrics Summary Based on Smoothed Curves ==========")
    print(metrics_df.round(4).to_string(index=False))

    print("\n========== Improvement Summary Based on Smoothed Curves ==========")
    print(improvement_df.round(2).to_string(index=False))

    print("\nSaved to:")
    print(metrics_csv)
    print(improvement_csv)
    print(metrics_tex)
    print(improvement_tex)

    return metrics_df, improvement_df, raw_curves, smoothed_curves


if __name__ == "__main__":
    # base
    # analyze_training_result_file(
    #     pkl_path="./algorithm_gather_comparison/cached_results/base_ep400_step50_seed100.pkl",
    #     summary_path=None,
    #     proposed_name="GLDVTO",
    #     output_dir="./analysis_output_base",
    #     tail_ratio=0.25,
    #     tol=0.05,
    #     stable_window=10,
    #     reward_alpha=0.9,
    #     latency_alpha=0.9,
    #     energy_alpha=0.95,
    # )

    # ablation
    analyze_training_result_file(
        pkl_path="./algorithm_gather_comparison/cached_results/ablation_ep400_step50_seed100.pkl",
        summary_path=None,
        proposed_name="GLDVTO",
        output_dir="./analysis_output_ablation",
        tail_ratio=0.25,
        tol=0.05,
        stable_window=10,
        reward_alpha=0.9,
        latency_alpha=0.9,
        energy_alpha=0.95,
    )