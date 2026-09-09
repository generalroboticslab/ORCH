#!/usr/bin/env python3
"""
Generate ablation study plots and CSVs comparing human, simple, critic, and no-critic methods.

Assumes logs are structured as:
results/logs/wildfire/
  level_name/
    seed/
      method_name/  (human, simple, critic, no-critic)
        data.csv

Outputs go into 'results/plots/ablation/'.
  - Score bar plots by level:      plots/ablation/{level}_max_score_bar.png
  - Cost bar plots by level:        plots/ablation/{level}_avg_cost_bar.png
  - Score over time line plots:     plots/ablation/{level}_score_over_time.png
  - Summary stats CSV:              plots/ablation/summary_stats.csv
  - Individual run results CSV:     plots/ablation/run_results.csv
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -- Configuration ------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))
LOGS_DIR   = os.path.join(RESULTS_DIR, "logs", "wildfire")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots", "ablation")

# Methods to compare (in order)
METHODS = ["human", "simple", "critic", "no-critic"]

# Pricing in dollars per 1 million tokens
COST_PER_MILLION_INPUT_TOKENS = 2.5
COST_PER_MILLION_OUTPUT_TOKENS = 10.0

# -- Data Collection ----------------------------------------------------------

def collect_data():
    """
    Collects data from logs/wildfire directory.
    Returns: dict[level][method][seed] = DataFrame
    """
    data = {}

    if not os.path.exists(LOGS_DIR):
        print(f"Warning: {LOGS_DIR} does not exist")
        return data

    for level in os.listdir(LOGS_DIR):
        level_path = os.path.join(LOGS_DIR, level)
        if not os.path.isdir(level_path):
            continue

        for seed in os.listdir(level_path):
            seed_path = os.path.join(level_path, seed)
            if not os.path.isdir(seed_path):
                continue

            for method in os.listdir(seed_path):
                method_path = os.path.join(seed_path, method)
                if not os.path.isdir(method_path):
                    continue

                # Only process known methods
                if method not in METHODS:
                    continue

                csv_fp = os.path.join(method_path, "data.csv")
                if not os.path.isfile(csv_fp):
                    print(f"Warning: Missing {csv_fp}")
                    continue

                try:
                    df = pd.read_csv(csv_fp)
                    # Store in nested dict structure
                    data.setdefault(level, {}).setdefault(method, {})[seed] = df
                except Exception as e:
                    print(f"Error reading {csv_fp}: {e}")
                    continue

    return data

def compute_max_score(df):
    """Extract maximum score achieved in trajectory."""
    if 'cumulative_score' in df.columns:
        return df['cumulative_score'].max()
    return 0

def compute_final_cost(df):
    """Calculate final cost from tokens or use cumulative_cost if available."""
    if 'cumulative_cost' in df.columns:
        return df['cumulative_cost'].iloc[-1]
    elif 'cumulative_input_tokens' in df.columns and 'cumulative_output_tokens' in df.columns:
        input_tokens = df['cumulative_input_tokens'].iloc[-1]
        output_tokens = df['cumulative_output_tokens'].iloc[-1]
        input_cost = (input_tokens / 1_000_000) * COST_PER_MILLION_INPUT_TOKENS
        output_cost = (output_tokens / 1_000_000) * COST_PER_MILLION_OUTPUT_TOKENS
        return input_cost + output_cost
    return 0

# -- Plotting Functions -------------------------------------------------------

def plot_score_bars(data):
    """Create bar plots showing max score by method for each level."""
    for level, methods_data in data.items():
        means, stds, labels = [], [], []

        for method in METHODS:
            if method not in methods_data:
                continue

            max_scores = [compute_max_score(df) for df in methods_data[method].values()]

            if len(max_scores) > 0:
                m = np.mean(max_scores)
                std = np.std(max_scores, ddof=1) if len(max_scores) > 1 else 0
                means.append(m)
                stds.append(std)
                labels.append(method)

        if len(labels) == 0:
            continue

        x = np.arange(len(labels))
        plt.figure(figsize=(8, 6))
        plt.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
        plt.xticks(x, labels, rotation=0)
        plt.title(f"{level} — Max Score by Method")
        plt.ylabel("Max Score")
        plt.xlabel("Method")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{level}_max_score_bar.png"), dpi=300)
        plt.close()

def plot_cost_bars(data):
    """Create bar plots showing average cost by method for each level."""
    for level, methods_data in data.items():
        means, stds, labels = [], [], []

        for method in METHODS:
            if method not in methods_data:
                continue

            costs = [compute_final_cost(df) for df in methods_data[method].values()]

            if len(costs) > 0:
                m = np.mean(costs)
                std = np.std(costs, ddof=1) if len(costs) > 1 else 0
                means.append(m)
                stds.append(std)
                labels.append(method)

        if len(labels) == 0:
            continue

        x = np.arange(len(labels))
        plt.figure(figsize=(8, 6))
        plt.bar(x, means, yerr=stds, capsize=5, alpha=0.8)
        plt.xticks(x, labels, rotation=0)
        plt.title(f"{level} — Average Cost by Method")
        plt.ylabel("Cost ($)")
        plt.xlabel("Method")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{level}_avg_cost_bar.png"), dpi=300)
        plt.close()

def plot_score_over_time(data):
    """Create line plots showing score over time for each level."""
    for level, methods_data in data.items():
        plt.figure(figsize=(10, 6))

        for method in METHODS:
            if method not in methods_data:
                continue

            dfs = list(methods_data[method].values())
            if len(dfs) == 0:
                continue

            # Find max length for padding
            max_len = max(len(df) for df in dfs)

            # Align trajectories
            aligned = []
            for df in dfs:
                if 'cumulative_score' not in df.columns:
                    continue
                scores = df['cumulative_score'].values
                max_score = scores.max()

                # Pad with max score if trajectory is shorter
                if len(scores) < max_len:
                    pad = np.full(max_len - len(scores), max_score)
                    scores = np.concatenate([scores, pad])
                else:
                    scores = scores[:max_len]

                aligned.append(scores)

            if len(aligned) == 0:
                continue

            # Compute statistics
            aligned_arr = np.vstack(aligned)
            mean_scores = aligned_arr.mean(axis=0)
            std_scores = aligned_arr.std(axis=0, ddof=1) if len(aligned) > 1 else np.zeros_like(mean_scores)

            # Plot
            x = np.arange(max_len)
            plt.plot(x, mean_scores, label=method, linewidth=2)
            plt.fill_between(x, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2)

        plt.title(f"{level} — Score Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Score")
        plt.legend()
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{level}_score_over_time.png"), dpi=300)
        plt.close()

# -- CSV Export Functions -----------------------------------------------------

def export_summary_stats(data):
    """Export summary statistics: level, method, mean_max_score, std_max_score, mean_cost, std_cost."""
    rows = []

    for level in sorted(data.keys()):
        methods_data = data[level]

        for method in METHODS:
            if method not in methods_data:
                continue

            dfs = list(methods_data[method].values())

            max_scores = [compute_max_score(df) for df in dfs]
            costs = [compute_final_cost(df) for df in dfs]

            mean_score = np.mean(max_scores)
            std_score = np.std(max_scores, ddof=1) if len(max_scores) > 1 else 0
            mean_cost = np.mean(costs)
            std_cost = np.std(costs, ddof=1) if len(costs) > 1 else 0

            rows.append({
                'level': level,
                'method': method,
                'mean_max_score': f"{mean_score:.2f}",
                'std_max_score': f"{std_score:.2f}",
                'mean_cost': f"{mean_cost:.2f}",
                'std_cost': f"{std_cost:.2f}"
            })

    df_summary = pd.DataFrame(rows)
    out_fp = os.path.join(OUTPUT_DIR, 'summary_stats.csv')
    df_summary.to_csv(out_fp, index=False)
    print(f"Saved summary stats to {out_fp}")

def export_run_results(data):
    """Export individual run results: level, seed, method, max_score, final_cost, run_length."""
    rows = []

    for level in sorted(data.keys()):
        methods_data = data[level]

        for method in METHODS:
            if method not in methods_data:
                continue

            for seed, df in methods_data[method].items():
                max_score = compute_max_score(df)
                final_cost = compute_final_cost(df)
                run_length = len(df)

                rows.append({
                    'level': level,
                    'seed': seed,
                    'method': method,
                    'max_score': f"{max_score:.2f}",
                    'final_cost': f"{final_cost:.2f}",
                    'run_length': run_length
                })

    df_results = pd.DataFrame(rows)
    out_fp = os.path.join(OUTPUT_DIR, 'run_results.csv')
    df_results.to_csv(out_fp, index=False)
    print(f"Saved run results to {out_fp}")

# -- Main ----------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Collecting data from {LOGS_DIR}...")
    data = collect_data()

    if not data:
        print(f"No data found under {LOGS_DIR}")
        return

    print(f"Found data for {len(data)} levels")
    for level, methods_data in data.items():
        print(f"  {level}: {list(methods_data.keys())}")

    print("\nGenerating plots...")
    plot_score_bars(data)
    plot_cost_bars(data)
    plot_score_over_time(data)

    print("\nExporting CSVs...")
    export_summary_stats(data)
    export_run_results(data)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
