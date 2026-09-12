import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

DATA_DIR = (
    Path(__file__).parent.parent
    / "algorithms"
    / "ORCH"
    / "results"
    / "experiment_data"
    / "FINAL"
)
NOOP_DIR = (
    Path(__file__).parent.parent
    / "algorithms"
    / "ORCH"
    / "results"
    / "experiment_data"
    / "ORCH_NOOP"
)
PLOT_DIR = Path(__file__).parent / "plots"

# Levels that get NOOP comparison plots
NOOP_LEVELS = {
    "Suppress_Fire_Contain", "Suppress_Fire_Contain_Water_Source",
    "Suppress_Fire_Extinguish", "Suppress_Fire_Extinguish_Second_Fire",
    "Suppress_Fire_Extinguish_Rapid_Growth",
    "Suppress_Fire_Locate_and_Suppress", "Suppress_Fire_Locate_Deploy_Suppress",
    "Full_Game",
}

# ── Per-level configuration ──────────────────────────────────────────────────
COMMON_METRICS = ["exploration", "cumulative_idle_steps", "cumulative_replans"]

_CUT = ["correct_trees_cut"]
_SCOUT = ["fire_scouted"]
_TRANSPORT = ["firefighters_transported"]
_RESCUE = ["civilians_rescued", "civilians_scouted"]
_FIRE = ["trees_fire", "fire_extinguished", "agents_destroyed"]
_FIRE_SCOUT = _FIRE + ["fire_scouted"]

LEVEL_CONFIGS = {
    "Cut_Trees_Sparse_small":  {"line_metrics": _CUT, "bar": {"metric": "correct_trees_cut", "label": "Correct Trees Cut"}},
    "Cut_Trees_Sparse_large":  {"line_metrics": _CUT, "bar": {"metric": "correct_trees_cut", "label": "Correct Trees Cut"}},
    "Cut_Trees_Lines_small":   {"line_metrics": _CUT, "bar": {"metric": "correct_trees_cut", "label": "Correct Trees Cut"}},
    "Cut_Trees_Lines_large":   {"line_metrics": _CUT, "bar": {"metric": "correct_trees_cut", "label": "Correct Trees Cut"}},
    "Scout_Fire_small":        {"line_metrics": _SCOUT, "bar": {"metric": "fire_scouted", "label": "Fire Scouted"}},
    "Scout_Fire_large":        {"line_metrics": _SCOUT, "bar": {"metric": "fire_scouted", "label": "Fire Scouted"}},
    "Scout_Fire_Drone_Lost":   {"line_metrics": _SCOUT, "bar": {"metric": "fire_scouted", "label": "Fire Scouted", "max_only": True}},
    "Transport_Firefighters_small": {"line_metrics": _TRANSPORT, "bar": {"metric": "firefighters_transported", "label": "Firefighters Transported"}},
    "Transport_Firefighters_large": {"line_metrics": _TRANSPORT, "bar": {"metric": "firefighters_transported", "label": "Firefighters Transported"}},
    "Transport_Helicopter_Down":    {"line_metrics": _TRANSPORT, "bar": {"metric": "firefighters_transported", "label": "Firefighters Transported", "max_only": True}},
    "Rescue_Civilians_Known_Location_small":  {"line_metrics": _RESCUE, "bar": {"metric": "civilians_rescued", "label": "Civilians Rescued"}},
    "Rescue_Civilians_Known_Location_large":  {"line_metrics": _RESCUE, "bar": {"metric": "civilians_rescued", "label": "Civilians Rescued"}},
    "Rescue_Civilians_Search_and_Rescue":     {"line_metrics": _RESCUE, "bar": {"metric": "civilians_rescued", "label": "Civilians Rescued"}},
    "Rescue_Civilians_Search_Rescue_Transport": {"line_metrics": _RESCUE, "bar": {"metric": "civilians_rescued", "label": "Civilians Rescued"}},
    "Rescue_Civilians_Surprise":              {"line_metrics": _RESCUE},
    "Suppress_Fire_Contain":                  {"line_metrics": _FIRE_SCOUT},
    "Suppress_Fire_Contain_Water_Source":      {"line_metrics": _FIRE_SCOUT},
    "Suppress_Fire_Extinguish":               {"line_metrics": _FIRE_SCOUT},
    "Suppress_Fire_Extinguish_Second_Fire":    {"line_metrics": _FIRE_SCOUT},
    "Suppress_Fire_Extinguish_Rapid_Growth":   {"line_metrics": _FIRE_SCOUT},
    "Suppress_Fire_Locate_and_Suppress":      {"line_metrics": _FIRE_SCOUT},
    "Suppress_Fire_Locate_Deploy_Suppress":   {"line_metrics": _FIRE_SCOUT},
}

LEVEL_DISPLAY_NAMES = {
    "Cut_Trees_Sparse_small": "Cut Trees: Sparse (Small)",
    "Cut_Trees_Sparse_large": "Cut Trees: Sparse (Large)",
    "Cut_Trees_Lines_small": "Cut Trees: Lines (Small)",
    "Cut_Trees_Lines_large": "Cut Trees: Lines (Large)",
    "Scout_Fire_small": "Scout Fire (Small)",
    "Scout_Fire_large": "Scout Fire (Large)",
    "Scout_Fire_Drone_Lost": "Scout Fire: Drone Lost",
    "Transport_Firefighters_small": "Transport Firefighters (Small)",
    "Transport_Firefighters_large": "Transport Firefighters (Large)",
    "Transport_Helicopter_Down": "Transport: Helicopter Down",
    "Rescue_Civilians_Known_Location_small": "Rescue Civilians: Known Location (Small)",
    "Rescue_Civilians_Known_Location_large": "Rescue Civilians: Known Location (Large)",
    "Rescue_Civilians_Search_and_Rescue": "Rescue Civilians: Search and Rescue",
    "Rescue_Civilians_Search_Rescue_Transport": "Rescue Civilians: Search + Rescue + Transport",
    "Rescue_Civilians_Surprise": "Rescue Civilians: Surprise",
    "Suppress_Fire_Contain": "Suppress Fire: Contain",
    "Suppress_Fire_Contain_Water_Source": "Suppress Fire: Contain (Water Source)",
    "Suppress_Fire_Extinguish": "Suppress Fire: Extinguish",
    "Suppress_Fire_Extinguish_Second_Fire": "Suppress Fire: Extinguish (Second Fire)",
    "Suppress_Fire_Extinguish_Rapid_Growth": "Suppress Fire: Extinguish (Rapid Growth)",
    "Suppress_Fire_Locate_and_Suppress": "Suppress Fire: Locate and Suppress",
    "Suppress_Fire_Locate_Deploy_Suppress": "Suppress Fire: Locate, Deploy, Suppress",
}

# ── Baseline comparison data (mean, std) and sample sizes ────────────────────
BASELINE_DATA = {
    "Cut_Trees_Sparse_small": {
        "max_score": 18,
        "CAMON": (18.00, 0.00), "COELA": (14.00, 4.18),
        "Embodied": (14.60, 2.07), "HMAS-2": (17.40, 0.89),
    },
    "Cut_Trees_Sparse_large": {
        "max_score": 75,
        "CAMON": (72.00, 4.00), "COELA": (57.67, 25.00),
        "Embodied": (50.33, 9.00), "HMAS-2": (56.33, 13.00),
    },
    "Cut_Trees_Lines_small": {
        "max_score": 30,
        "CAMON": (30.00, 0.00), "COELA": (30.00, 0.00),
        "Embodied": (30.00, 0.00), "HMAS-2": (28.00, 2.65),
    },
    "Cut_Trees_Lines_large": {
        "max_score": 105,
        "CAMON": (94.33, 7.00), "COELA": (83.67, 24.00),
        "Embodied": (90.33, 1.00), "HMAS-2": (90.33, 6.00),
    },
    "Scout_Fire_small": {
        "max_score": 2,
        "CAMON": (1.60, 0.89), "COELA": (0.00, 0.00),
        "Embodied": (0.80, 0.84), "HMAS-2": (1.20, 1.10),
    },
    "Scout_Fire_large": {
        "max_score": 2,
        "CAMON": (0.00, 0.00), "COELA": (0.00, 0.00),
        "Embodied": (0.00, 0.00), "HMAS-2": (0.00, 0.00),
    },
    "Scout_Fire_Drone_Lost": {"max_score": 2},
    "Transport_Firefighters_small": {
        "max_score": 6,
        "CAMON": (6.00, 0.00), "COELA": (3.00, 5.00),
        "Embodied": (4.60, 4.00), "HMAS-2": (4.40, 5.00),
    },
    "Transport_Firefighters_large": {
        "max_score": 12,
        "CAMON": (10.00, 0.00), "COELA": (8.33, 5.00),
        "Embodied": (9.33, 2.00), "HMAS-2": (8.33, 5.00),
    },
    "Transport_Helicopter_Down": {"max_score": 10},
    "Rescue_Civilians_Known_Location_small": {
        "max_score": 3,
        "CAMON": (3.00, 0.00), "COELA": (0.67, 0.58),
        "Embodied": (2.67, 0.58), "HMAS-2": (2.33, 0.58),
    },
    "Rescue_Civilians_Known_Location_large": {
        "max_score": 9,
        "CAMON": (4.00, 5.00), "COELA": (0.00, 0.00),
        "Embodied": (4.33, 3.00), "HMAS-2": (4.00, 2.00),
    },
    "Rescue_Civilians_Search_and_Rescue": {
        "max_score": 5,
        "CAMON": (0.00, 0.00), "COELA": (0.00, 0.00),
        "Embodied": (0.00, 0.00), "HMAS-2": (0.00, 0.00),
    },
    "Rescue_Civilians_Search_Rescue_Transport": {
        "max_score": 10,
        "CAMON": (0.00, 0.00), "COELA": (0.00, 0.00),
        "Embodied": (0.00, 0.00), "HMAS-2": (0.00, 0.00),
    },
}

BASELINE_N = {
    "Cut_Trees_Sparse_small": 10, "Cut_Trees_Sparse_large": 5,
    "Cut_Trees_Lines_small": 6, "Cut_Trees_Lines_large": 3,
    "Scout_Fire_small": 10, "Scout_Fire_large": 5,
    "Transport_Firefighters_small": 5, "Transport_Firefighters_large": 5,
    "Rescue_Civilians_Known_Location_small": 8, "Rescue_Civilians_Known_Location_large": 5,
    "Rescue_Civilians_Search_and_Rescue": 3, "Rescue_Civilians_Search_Rescue_Transport": 3,
}


def load_all_runs():
    """Walk FINAL/ and load latest data.csv per seed, tagged by level."""
    runs = defaultdict(list)
    for level_dir in sorted(DATA_DIR.iterdir()):
        if not level_dir.is_dir() or level_dir.name == "extra":
            continue
        level = level_dir.name
        for seed_dir in sorted(level_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            preset_dir = seed_dir / "preset"
            if not preset_dir.is_dir():
                continue
            ts_dirs = sorted([d for d in preset_dir.iterdir() if d.is_dir()])
            if not ts_dirs:
                continue
            latest = ts_dirs[-1]
            csv_path = latest / "data.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                runs[level].append(df)
    return runs


def load_noop_runs():
    """Walk ORCH_NOOP/ and load latest data.csv per seed, tagged by level."""
    runs = defaultdict(list)
    if not NOOP_DIR.exists():
        return runs
    for level_dir in sorted(NOOP_DIR.iterdir()):
        if not level_dir.is_dir():
            continue
        level = level_dir.name
        for seed_dir in sorted(level_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            # NOOP structure: LEVEL/SEED/TIMESTAMP/data.csv (no preset/)
            ts_dirs = sorted([d for d in seed_dir.iterdir() if d.is_dir()])
            if not ts_dirs:
                continue
            latest = ts_dirs[-1]
            csv_path = latest / "data.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                runs[level].append(df)
    return runs


NON_MONOTONIC_METRICS = {"trees_fire", "agents_destroyed"}


def compute_timestep_stats(dfs, metric):
    """Compute mean and SEM of a metric at each timestep across runs."""
    max_t = int(max(df["timestep"].max() for df in dfs))
    all_vals = []
    for df in dfs:
        series = df.set_index("timestep")[metric].reindex(range(max_t + 1))
        series = series.ffill()
        series = series.bfill()
        if metric not in NON_MONOTONIC_METRICS:
            series = series.cummax()
        all_vals.append(series.values)
    matrix = np.array(all_vals)
    means = matrix.mean(axis=0)
    sems = (
        matrix.std(axis=0, ddof=1) / np.sqrt(matrix.shape[0])
        if matrix.shape[0] > 1
        else np.zeros(max_t + 1)
    )
    return np.arange(max_t + 1), means, sems


def compute_idle_replan_stats(dfs):
    """Compute average idle steps per timestep and timesteps per replan."""
    idle_per_step_vals = []
    steps_per_replan_vals = []
    for df in dfs:
        final = df.iloc[-1]
        n_steps = final["timestep"]
        total_idle = final["cumulative_idle_steps"]
        total_replans = final["cumulative_replans"]
        if n_steps > 0:
            idle_per_step_vals.append(total_idle / n_steps)
        if total_replans > 0:
            steps_per_replan_vals.append(n_steps / total_replans)
    avg_idle = np.mean(idle_per_step_vals) if idle_per_step_vals else 0.0
    avg_spr = np.mean(steps_per_replan_vals) if steps_per_replan_vals else float("inf")
    return avg_idle, avg_spr


def plot_line(dfs, level, metric, ylabel, subtitle=None):
    """Line plot with SEM shading for a single metric."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ts, mean, sem = compute_timestep_stats(dfs, metric)
    ax.plot(ts, mean, color="#1f77b4", linewidth=2)
    ax.fill_between(ts, mean - sem, mean + sem, color="#1f77b4", alpha=0.2)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    display = LEVEL_DISPLAY_NAMES.get(level, level.replace("_", " "))
    title = f"{display} — {ylabel} over Time"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def welch_ttest_from_stats(mean1, std1, n1, mean2, std2, n2):
    """Welch's t-test from summary statistics. Returns p-value."""
    if n1 < 2 or n2 < 2:
        return np.nan
    se1 = std1**2 / n1
    se2 = std2**2 / n2
    se_sum = se1 + se2
    if se_sum == 0:
        return 1.0 if mean1 == mean2 else np.nan
    t_stat = (mean1 - mean2) / np.sqrt(se_sum)
    df = se_sum**2 / (se1**2 / (n1 - 1) + se2**2 / (n2 - 1)) if (se1 > 0 or se2 > 0) else 1
    return stats.t.sf(abs(t_stat), df) * 2


def significance_str(p):
    """Convert p-value to significance annotation."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return f"p={p:.2f}"


def plot_bar_comparison(dfs, level, metric, ylabel, max_only=False):
    """Bar chart comparing our final score to baselines with error bars and p-values."""
    final_vals = [df[metric].iloc[-1] for df in dfs]
    ours_mean = np.mean(final_vals)
    ours_std = np.std(final_vals, ddof=1) if len(final_vals) > 1 else 0.0
    ours_n = len(final_vals)

    baselines = BASELINE_DATA[level]
    max_score = baselines["max_score"]
    baseline_n = BASELINE_N.get(level, 5)

    if max_only:
        labels = ["Ours"]
        means = np.array([ours_mean])
        stds = np.array([ours_std])
        colors = ["#1f77b4"]
    else:
        baseline_keys = ["CAMON", "COELA", "Embodied", "HMAS-2"]
        labels = ["Ours"] + baseline_keys
        means = np.array([ours_mean] + [baselines[k][0] for k in baseline_keys])
        stds = np.array([ours_std] + [baselines[k][1] for k in baseline_keys])
        colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]

    lower_err = np.minimum(stds, means)
    upper_err = np.minimum(stds, max_score - means)
    upper_err = np.maximum(upper_err, 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors, yerr=[lower_err, upper_err], capsize=5, ecolor="black")
    ax.axhline(y=max_score, color="black", linestyle=":", linewidth=1.5, label=f"Max ({max_score})")
    ax.legend()

    # Add p-value annotations for baselines
    if not max_only:
        for i, key in enumerate(baseline_keys):
            b_mean, b_std = baselines[key]
            p = welch_ttest_from_stats(ours_mean, ours_std, ours_n, b_mean, b_std, baseline_n)
            sig = significance_str(p)
            if sig:
                bar_top = means[i + 1] + upper_err[i + 1]
                ax.text(
                    x[i + 1], bar_top + max_score * 0.02, sig,
                    ha="center", va="bottom", fontsize=8, color="black",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    display = LEVEL_DISPLAY_NAMES.get(level, level.replace("_", " "))
    ax.set_title(f"{display} — Final {ylabel} Comparison")
    ax.set_ylim(0, max_score * 1.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_radar(dfs, level, cfg, noop_dfs=None):
    """Radar plot showing final values of all evaluation dimensions."""
    if noop_dfs is None:
        noop_dfs = []
    # Build list of (label, per-run values) for each axis
    axes = []

    # Exploration
    exploration_vals = [df["exploration"].iloc[-1] for df in dfs]
    axes.append(("Exploration", exploration_vals, False))

    # Idle per step (lower is better)
    idle_vals = []
    for df in dfs:
        final = df.iloc[-1]
        n_steps = final["timestep"]
        idle_vals.append(final["cumulative_idle_steps"] / n_steps if n_steps > 0 else 0)
    axes.append(("Idle/Step", idle_vals, True))

    # Steps per replan
    spr_vals = []
    for df in dfs:
        final = df.iloc[-1]
        replans = final["cumulative_replans"]
        spr_vals.append(final["timestep"] / replans if replans > 0 else 0)
    axes.append(("Steps/Replan", spr_vals, False))

    # Level-specific metrics
    all_metrics = list(cfg["line_metrics"])
    bar = cfg.get("bar")
    if bar and bar["metric"] not in all_metrics:
        all_metrics.append(bar["metric"])
    INVERTED_METRICS = {"agents_destroyed"}
    for metric in all_metrics:
        if metric == "trees_fire":
            continue  # replaced by trees_saved below
        vals = [df[metric].iloc[-1] for df in dfs]
        label = metric.replace("_", " ").title()
        axes.append((label, vals, metric in INVERTED_METRICS))

    # Trees Saved: noop trees_fire - ours trees_fire (positive = better)
    if "trees_fire" in all_metrics and noop_dfs:
        noop_final_mean = np.mean([df["trees_fire"].iloc[-1] for df in noop_dfs])
        saved_vals = [noop_final_mean - df["trees_fire"].iloc[-1] for df in dfs]
        axes.append(("Trees Saved", saved_vals, False))
    elif "trees_fire" in all_metrics:
        # No noop data: fall back to inverted trees_fire
        vals = [df["trees_fire"].iloc[-1] for df in dfs]
        axes.append(("Trees Fire", vals, True))

    # Normalize each axis to [0, 1]
    n_axes = len(axes)
    norm_means = []
    norm_stds = []
    labels = []
    for label, vals, invert in axes:
        arr = np.array(vals, dtype=float)
        max_val = arr.max()
        if max_val > 0:
            normed = arr / max_val
        else:
            normed = np.zeros_like(arr)
        if invert:
            normed = 1.0 - normed
        norm_means.append(normed.mean())
        norm_stds.append(normed.std(ddof=1) if len(normed) > 1 else 0.0)
        labels.append(label)

    # Build radar
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]

    mean_vals = norm_means + [norm_means[0]]
    upper_vals = [min(m + s, 1.0) for m, s in zip(norm_means, norm_stds)] + [min(norm_means[0] + norm_stds[0], 1.0)]
    lower_vals = [max(m - s, 0.0) for m, s in zip(norm_means, norm_stds)] + [max(norm_means[0] - norm_stds[0], 0.0)]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, mean_vals, color="#1f77b4", linewidth=2, marker="o", markersize=6)
    ax.fill_between(angles, lower_vals, upper_vals, color="#1f77b4", alpha=0.15)
    ax.fill(angles, mean_vals, color="#1f77b4", alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.grid(color="gray", linestyle="--", linewidth=0.5)

    display = LEVEL_DISPLAY_NAMES.get(level, level.replace("_", " "))
    ax.set_title(f"{display} — Evaluation Radar", y=1.08, fontsize=12)
    fig.tight_layout()
    return fig


def compute_mean_series(dfs, metric):
    """Compute mean of a metric at each timestep, forward-filling shorter runs."""
    max_t = int(max(df["timestep"].max() for df in dfs))
    all_vals = []
    for df in dfs:
        series = df.set_index("timestep")[metric].reindex(range(max_t + 1))
        series = series.ffill().bfill()
        all_vals.append(series.values)
    matrix = np.array(all_vals)
    return np.arange(max_t + 1), matrix.mean(axis=0)


def plot_noop_trees_saved(ours_dfs, noop_dfs, level):
    """Line plot: NOOP trees_fire vs Ours trees_fire, showing trees saved."""
    _, noop_ts_mean = compute_mean_series(noop_dfs, "trees_fire")
    _, ours_ts_mean = compute_mean_series(ours_dfs, "trees_fire")

    # Align to the longer series
    max_len = max(len(noop_ts_mean), len(ours_ts_mean))
    if len(noop_ts_mean) < max_len:
        noop_ts_mean = np.pad(noop_ts_mean, (0, max_len - len(noop_ts_mean)),
                              constant_values=noop_ts_mean[-1])
    if len(ours_ts_mean) < max_len:
        ours_ts_mean = np.pad(ours_ts_mean, (0, max_len - len(ours_ts_mean)),
                              constant_values=ours_ts_mean[-1])

    trees_saved = noop_ts_mean[-1] - ours_ts_mean[-1]
    ts = np.arange(max_len)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, noop_ts_mean, color="#d62728", linewidth=2, label="No-Op")
    ax.plot(ts, ours_ts_mean, color="#1f77b4", linewidth=2, label="Ours")
    ax.fill_between(ts, ours_ts_mean, noop_ts_mean,
                    where=noop_ts_mean >= ours_ts_mean,
                    color="#2ca02c", alpha=0.15, label="Trees Saved")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Trees on Fire")
    ax.legend()
    display = LEVEL_DISPLAY_NAMES.get(level, level.replace("_", " "))
    ax.set_title(f"{display} — Trees on Fire: Ours vs No-Op\nTrees Saved: {trees_saved:.1f}",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_noop_bar(ours_dfs, noop_dfs, level):
    """Bar chart: final trees_fire for No-Op vs Ours."""
    noop_final = np.mean([df["trees_fire"].iloc[-1] for df in noop_dfs])
    ours_final = np.mean([df["trees_fire"].iloc[-1] for df in ours_dfs])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["No-Op", "Ours"], [noop_final, ours_final],
           color=["#d62728", "#1f77b4"])
    ax.set_ylabel("Final Trees on Fire")
    display = LEVEL_DISPLAY_NAMES.get(level, level.replace("_", " "))
    ax.set_title(f"{display} — Final Trees on Fire")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    PLOT_DIR.mkdir(exist_ok=True)
    runs = load_all_runs()
    noop_runs = load_noop_runs()

    for level, dfs in sorted(runs.items()):
        if not dfs:
            continue
        cfg = LEVEL_CONFIGS.get(level)
        if cfg is None:
            print(f"No config for {level}, skipping")
            continue

        avg_idle, avg_spr = compute_idle_replan_stats(dfs)

        all_metrics = list(COMMON_METRICS) + cfg["line_metrics"]
        bar = cfg.get("bar")
        if bar and bar["metric"] not in all_metrics:
            all_metrics.append(bar["metric"])
        eval_list = ", ".join(m.replace("_", " ") for m in all_metrics)

        print(f"\n{level}: {len(dfs)} runs")
        print(f"  Avg idle/step: {avg_idle:.2f} | Steps/replan: {avg_spr:.1f}")
        print(f"  Evaluations: {eval_list}")

        idle_subtitle = f"Avg idle per step: {avg_idle:.2f}"
        replan_subtitle = f"Avg steps per replan: {avg_spr:.1f}"

        # Common line plots
        for metric in COMMON_METRICS:
            ylabel = metric.replace("_", " ").title()
            sub = None
            if metric == "cumulative_idle_steps":
                sub = idle_subtitle
            elif metric == "cumulative_replans":
                sub = replan_subtitle
            fig = plot_line(dfs, level, metric, ylabel, subtitle=sub)
            fig.savefig(PLOT_DIR / f"{level}_{metric}_over_time.png", dpi=150)
            plt.close(fig)

        # Level-specific line metrics
        for metric in cfg["line_metrics"]:
            ylabel = metric.replace("_", " ").title()
            fig = plot_line(dfs, level, metric, ylabel)
            fig.savefig(PLOT_DIR / f"{level}_{metric}_over_time.png", dpi=150)
            plt.close(fig)

        # Bar chart
        if bar:
            fig = plot_bar_comparison(
                dfs, level, bar["metric"], bar["label"],
                max_only=bar.get("max_only", False),
            )
            fig.savefig(PLOT_DIR / f"{level}_final_{bar['metric']}_bar.png", dpi=150)
            plt.close(fig)

        # Radar plot
        fig = plot_radar(dfs, level, cfg, noop_dfs=noop_runs.get(level, []))
        fig.savefig(PLOT_DIR / f"{level}_radar.png", dpi=150)
        plt.close(fig)

    # ── NOOP comparison plots for Suppress Fire / Full Game ─────────────
    for level in sorted(NOOP_LEVELS):
        ours_dfs = runs.get(level, [])
        noop_dfs = noop_runs.get(level, [])
        if not ours_dfs or not noop_dfs:
            continue
        print(f"\n{level}: NOOP comparison ({len(noop_dfs)} noop runs)")

        fig = plot_noop_trees_saved(ours_dfs, noop_dfs, level)
        fig.savefig(PLOT_DIR / f"{level}_trees_saved_vs_noop.png", dpi=150)
        plt.close(fig)

        fig = plot_noop_bar(ours_dfs, noop_dfs, level)
        fig.savefig(PLOT_DIR / f"{level}_final_trees_fire_noop_bar.png", dpi=150)
        plt.close(fig)

    # ── Cross-level summary bar charts ──────────────────────────────────
    level_labels = []
    idle_vals = []
    spr_vals = []
    for level, dfs in sorted(runs.items()):
        if not dfs or level not in LEVEL_CONFIGS:
            continue
        avg_idle, avg_spr = compute_idle_replan_stats(dfs)
        display = LEVEL_DISPLAY_NAMES.get(level, level.replace("_", " "))
        level_labels.append(display)
        idle_vals.append(avg_idle)
        spr_vals.append(avg_spr)

    x = np.arange(len(level_labels))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, idle_vals, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Avg Idle Steps per Timestep")
    ax.set_title("Average Idle per Step Across Levels")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "all_levels_idle_per_step.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x, spr_vals, color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(level_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Avg Steps per Replan")
    ax.set_title("Average Steps per Replan Across Levels")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "all_levels_steps_per_replan.png", dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
