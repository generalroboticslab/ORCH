#!/usr/bin/env python3
"""
Calculate and plot Behavioral Competency Score (BCS) for all algorithms.
- Loads data from Logs/ in the same way as data_analysis.py
- Does NOT require all seeds/algorithms to be present (just warns if missing)
- Exports BCS table as CSV and radar chart as PNG
"""
import os
import pandas as pd
import numpy as np
from crew_algorithms.wildfire_alg.data.radar import save_highdef_radar

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR   = os.path.join(BASE_DIR, "Logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")

BEHAVIORAL_GOALS = [
    ("TD", "Task Decomposition"),
    ("AC", "Adaptive Coordination"),
    ("SR", "Search and Rescue"),
    ("OS", "Open-ended Suppression"),
    ("RC", "Real-time Communication"),
    ("PA", "Planning and Allocation"),
]
LEVEL_TO_GOALS = {
    'Transport_Firefighters_small': ["RC", "TD"],
    'Transport_Firefighters_large': ["RC", "TD"],
    'Rescue_Civilians_Known_Location_small': ["SR", "TD"],
    'Rescue_Civilians_Known_Location_large': ["SR", "TD"],
    'Rescue_Civilians_Search_and_Rescue': ["SR", "AC"],
    'Rescue_Civilians_Search_Rescue_Transport': ["SR", "AC", "PA"],
    'Suppress_Fire_Contain': ["OS", "AC"],
    'Suppress_Fire_Extinguish': ["OS", "AC"],
    'Suppress_Fire_Locate_and_Suppress': ["OS", "AC", "PA"],
    'Suppress_Fire_Locate_Deploy_Suppress': ["OS", "AC", "PA"],
    'Cut_Trees_Sparse_small': ["TD", "PA"],
    'Cut_Trees_Sparse_large': ["TD", "PA"],
    'Cut_Trees_Lines_small': ["TD", "PA"],
    'Cut_Trees_Lines_large': ["TD", "PA"],
    'Scout_Fire_small': ["RC", "AC"],
    'Scout_Fire_large': ["RC", "AC"],
    'Full_Game': ["TD", "AC", "SR", "OS", "RC", "PA"],
}

def get_latest_timestamp(seed_path):
    ts = [d for d in os.listdir(seed_path)
          if os.path.isdir(os.path.join(seed_path, d))]
    return max(ts) if ts else None

def collect_data():
    data = {}
    if not os.path.isdir(LOGS_DIR):
        print(f"No Logs directory at {LOGS_DIR}")
        return data
    for algo in os.listdir(LOGS_DIR):
        algo_path = os.path.join(LOGS_DIR, algo)
        if not os.path.isdir(algo_path):
            continue
        for level in os.listdir(algo_path):
            level_path = os.path.join(algo_path, level)
            if not os.path.isdir(level_path):
                continue
            for seed in os.listdir(level_path):
                seed_path = os.path.join(level_path, seed)
                if not os.path.isdir(seed_path):
                    continue
                ts = get_latest_timestamp(seed_path)
                if not ts:
                    continue
                csv_fp = os.path.join(seed_path, ts, "action_reward.csv")
                if not os.path.isfile(csv_fp):
                    continue
                df = pd.read_csv(csv_fp)
                # Attach seed info for later matching
                df.attrs['seed'] = seed
                data.setdefault(level, {}).setdefault(algo, []).append(df)
    return data

def get_target(level, data):
    if level.startswith('Cut_Trees'):
        for algo in data.get(level, {}):
            for df in data[level][algo]:
                if 'cumulative_score' in df:
                    return df['cumulative_score'].max()
        return 1
    if level.startswith('Transport_Firefighters'):
        if 'small' in level:
            return 6
        if 'large' in level:
            return 12
    if level.startswith('Scout_Fire'):
        return 2
    if level.startswith('Rescue_Civilians_Known_Location'):
        if 'small' in level:
            return 3
        if 'large' in level:
            return 9
    if level.startswith('Rescue_Civilians_Search_and_Rescue'):
        return 5
    if level.startswith('Rescue_Civilians_Search_Rescue_Transport'):
        return 10
    if level.startswith('Suppress_Fire') or level == 'Full_Game':
        return 0
    return 1

def get_type(level):
    if level.startswith('Suppress_Fire') or level == 'Full_Game':
        return 'open'
    return 'finite'

def export_bcs_stats(data):
    algos = sorted({algo for lvl in data for algo in data[lvl]})
    levels = [lvl for lvl in data if lvl in LEVEL_TO_GOALS]
    seeds_by_algo_level = {lvl: {algo: set() for algo in algos} for lvl in levels}
    for lvl in levels:
        for algo in algos:
            dfs = data[lvl].get(algo, [])
            for df in dfs:
                if 'seed' in df.attrs:
                    seeds_by_algo_level[lvl][algo].add(df.attrs['seed'])
                elif 'seed' in df.columns:
                    seeds_by_algo_level[lvl][algo].add(df['seed'].iloc[0])
                else:
                    seeds_by_algo_level[lvl][algo].add(str(dfs.index(df)))
    # Use all available seeds (no intersection required)
    norm_scores = {lvl: {algo: [] for algo in algos} for lvl in levels}
    for lvl in levels:
        target = get_target(lvl, data)
        typ = get_type(lvl)
        all_seeds = set()
        for algo in algos:
            all_seeds.update(seeds_by_algo_level[lvl][algo])
        for seed in all_seeds:
            # Baseline: DO-NOTHING
            do_nothing_dfs = data[lvl].get('DO-NOTHING', [])
            baseline = None
            for df in do_nothing_dfs:
                if 'seed' in df.attrs and str(df.attrs['seed']) == str(seed):
                    baseline = df['cumulative_score'].iloc[-1]
                    break
                elif 'seed' in df.columns and str(df['seed'].iloc[0]) == str(seed):
                    baseline = df['cumulative_score'].iloc[-1]
                    break
                elif str(do_nothing_dfs.index(df)) == str(seed):
                    baseline = df['cumulative_score'].iloc[-1]
                    break
            if baseline is None:
                print(f"[Warning] Missing DO-NOTHING data for {lvl} seed {seed}. Skipping this seed for normalization.")
                continue
            for algo in algos:
                dfs = data[lvl].get(algo, [])
                found = False
                for df in dfs:
                    if 'seed' in df.attrs and str(df.attrs['seed']) == str(seed):
                        sa = df['cumulative_score'].iloc[-1]
                        found = True
                        break
                    elif 'seed' in df.columns and str(df['seed'].iloc[0]) == str(seed):
                        sa = df['cumulative_score'].iloc[-1]
                        found = True
                        break
                    elif str(dfs.index(df)) == str(seed):
                        sa = df['cumulative_score'].iloc[-1]
                        found = True
                        break
                if not found:
                    print(f"[Warning] Missing data for {algo} on {lvl} seed {seed}. Skipping this seed for this algorithm.")
                    continue
                if typ == 'finite':
                    ns = (sa - baseline) / (target - baseline) if (target - baseline) != 0 else 0
                else:
                    ns = np.log1p((sa - baseline) / (target - baseline)) / np.log(2) if (target - baseline) != 0 else 0
                ns = max(0, min(1, ns))
                norm_scores[lvl][algo].append(ns)
    goal_to_levels = {abbr: [lvl for lvl, goals in LEVEL_TO_GOALS.items() if abbr in goals] for abbr, _ in BEHAVIORAL_GOALS}
    bcs = {abbr: {algo: None for algo in algos} for abbr, _ in BEHAVIORAL_GOALS}
    for abbr, _ in BEHAVIORAL_GOALS:
        for algo in algos:
            vals = []
            for lvl in goal_to_levels[abbr]:
                vals.extend(norm_scores[lvl][algo])
            bcs[abbr][algo] = np.mean(vals) if vals else 0.0
    bcs_df = pd.DataFrame({algo: [bcs[abbr][algo] for abbr, _ in BEHAVIORAL_GOALS] for algo in algos},
                         index=[abbr for abbr, _ in BEHAVIORAL_GOALS])
    bcs_df.index.name = 'Behaviour'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_fp = os.path.join(OUTPUT_DIR, 'bcs_stats.csv')
    bcs_df.to_csv(out_fp)
    print(f"Saved BCS stats to {out_fp}")
    radar_df = bcs_df.reset_index()
    save_highdef_radar(radar_df, 'Behavioral Competency Score (BCS)', os.path.join(OUTPUT_DIR, 'bcs_radar.png'), dpi=300, figsize=(10,10))

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = collect_data()
    if not data:
        print(f"No data under {LOGS_DIR}")
        return
    export_bcs_stats(data)
    print("BCS stats and radar chart generated in", OUTPUT_DIR)

if __name__ == '__main__':
    main() 