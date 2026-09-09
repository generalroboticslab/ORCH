#!/usr/bin/env python3
"""
Calculate and plot Behavioral Competency Score (BCS) from CSV data.
Uses mean scores directly from CSV instead of loading from Logs/.
"""
import os
import pandas as pd
import numpy as np
from crew_algorithms.wildfire_alg.data.radar import save_highdef_radar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    'Transport_Firefighters_small': ["AC", "SR", "RC"],
    'Transport_Firefighters_large': ["AC", "SR", "RC"],
    'Rescue_Civilians_Known_Location_small': ["TD", "SR", "PA"],
    'Rescue_Civilians_Known_Location_large': ["TD", "SR", "PA"],
    'Rescue_Civilians_Search_and_Rescue': ["TD", "SR", "PA", "OS"],
    'Rescue_Civilians_Search_Rescue_Transport': ["TD", "SR", "PA", "OS", "AC", "RC"],
    'Suppress_Fire_Contain': ["TD", "SR", "PA", "AC"],
    'Suppress_Fire_Extinguish': ["TD", "SR", "PA"],
    'Suppress_Fire_Locate_and_Suppress': ["TD", "SR", "PA", "AC", "OS"],
    'Suppress_Fire_Locate_Transport_Suppress': ["TD", "SR", "PA", "AC", "OS", "RC"],
    'Cut_Trees_Sparse_small': ["TD"],
    'Cut_Trees_Sparse_large': ["TD"],
    'Cut_Trees_Lines_small': ["TD", "AC"],
    'Cut_Trees_Lines_large': ["TD", "AC"],
    'Scout_Fire_small': ["TD", "SR", "OS"],
    'Scout_Fire_large': ["TD", "SR", "OS"],
    'Full_Environment': ["TD", "SR", "PA", "AC", "OS", "RC"],
}

CSV_DATA = """level,max_score,CAMON,COELA,Embodied,HMAS_2,Do_Nothing
Cut_Trees_Sparse_small,18,"18.00±0.00","14.00±4.18","14.60±2.07","17.40±0.89","0.00±0.00"
Cut_Trees_Sparse_large,75,"72.00±4.00","57.67±25.00","50.33±9.00","56.33±13.00","0.00±0.00"
Cut_Trees_Lines_small,30,"30.00±0.00","30.00±0.00","30.00±0.00","28.00±2.65","0.00±0.00"
Cut_Trees_Lines_large,105,"94.33±7.00","83.67±24.00","90.33±1.00","90.33±6.00","0.00±0.00"
Scout_Fire_small,2,"1.60±0.89","0.00±0.00","0.80±0.84","1.20±1.10","0.00±0.00"
Scout_Fire_large,2,"0.00±0.00","0.00±0.00","0.00±0.00","0.00±0.00","0.00±0.00"
Transport_Firefighters_small,6,"6.00±0.00","3.00±5.00","4.60±4.00","4.40±5.00","0.00±0.00"
Transport_Firefighters_large,12,"10.00±0.00","8.33±5.00","9.33±2.00","8.33±5.00","0.00±0.00"
Rescue_Civilians_Known_Location_small,3,"3.00±0.00","0.67±0.58","2.67±0.58","2.33±0.58","0.00±0.00"
Rescue_Civilians_Known_Location_large,9,"4.00±5.00","0.00±0.00","4.33±3.00","4.00±2.00","0.00±0.00"
Rescue_Civilians_Search_and_Rescue,5,"0.00±0.00","0.00±0.00","0.00±0.00","0.00±0.00","0.00±0.00"
Rescue_Civilians_Search_Rescue_Transport,10,"0.00±0.00","0.00±0.00","0.00±0.00","0.00±0.00","0.00±0.00"
Suppress_Fire_Extinguish,0,"-593.33±196.51","-635.00±189.65","-636.33±193.67","-624.33±196.92","-679.67±174.91"
Suppress_Fire_Contain,0,"-736.67±200.04","-751.33±202.60","-757.33±177.42","-739.00±199.96","-780.67±184.50"
Suppress_Fire_Locate_and_Suppress,0,"-1062.67±248.61","-1062.67±248.61","-1062.67±248.61","-1062.67±248.61","-1182.67±248.61"
Suppress_Fire_Locate_Transport_Suppress,0,"-729.67±387.07","-729.67±387.07","-729.67±387.07","-729.67±387.07","-929.67±387.07"
Full_Environment,0,"-5571.67±3347.06","-5516.00±3348.81","-5539.33±3381.69","-5522.67±3358.20","-5722.67±3330.02"
"""

def parse_mean_from_string(value_str):
    """Extract mean from 'mean±sd' format, ignoring SD."""
    if isinstance(value_str, (int, float)):
        return float(value_str)
    value_str = str(value_str).strip().strip('"')
    if '±' in value_str:
        return float(value_str.split('±')[0])
    return float(value_str)

def get_type(level):
    """Determine if level is finite-horizon or open-ended."""
    if level.startswith('Suppress_Fire') or level == 'Full_Environment':
        return 'open'
    return 'finite'

def load_data():
    """Load and parse CSV data."""
    from io import StringIO
    df = pd.read_csv(StringIO(CSV_DATA))

    # Parse mean values from each column (ignoring SD)
    for col in df.columns:
        if col not in ['level', 'max_score']:
            df[col] = df[col].apply(parse_mean_from_string)

    df['max_score'] = df['max_score'].astype(float)
    return df

def calculate_bcs(df):
    """Calculate BCS scores for each algorithm."""
    # Get algorithm names (exclude level, max_score, Do_Nothing)
    algos = [col for col in df.columns if col not in ['level', 'max_score', 'Do_Nothing']]

    # Filter to only levels in LEVEL_TO_GOALS
    df_filtered = df[df['level'].isin(LEVEL_TO_GOALS.keys())].copy()

    # Normalize scores for each level
    norm_scores = {lvl: {algo: None for algo in algos} for lvl in df_filtered['level']}

    for idx, row in df_filtered.iterrows():
        level = row['level']
        target = row['max_score']
        baseline = row['Do_Nothing']
        typ = get_type(level)

        for algo in algos:
            score = row[algo]

            # Normalize based on task type
            if typ == 'finite':
                # Finite-horizon: linear normalization
                if (target - baseline) != 0:
                    ns = (score - baseline) / (target - baseline)
                else:
                    ns = 0
            else:
                # Open-ended: logarithmic normalization
                if (target - baseline) != 0:
                    ns = np.log1p((score - baseline) / (target - baseline)) / np.log(2)
                else:
                    ns = 0

            # Clamp to [0, 1]
            ns = max(0, min(1, ns))
            norm_scores[level][algo] = ns

    # Map behavioral goals to levels
    goal_to_levels = {
        abbr: [lvl for lvl, goals in LEVEL_TO_GOALS.items() if abbr in goals]
        for abbr, _ in BEHAVIORAL_GOALS
    }

    # Calculate BCS for each behavioral goal and algorithm
    bcs = {abbr: {algo: None for algo in algos} for abbr, _ in BEHAVIORAL_GOALS}

    for abbr, _ in BEHAVIORAL_GOALS:
        for algo in algos:
            vals = []
            for lvl in goal_to_levels[abbr]:
                if lvl in norm_scores and norm_scores[lvl][algo] is not None:
                    vals.append(norm_scores[lvl][algo])
            bcs[abbr][algo] = np.mean(vals) if vals else 0.0

    # Create BCS DataFrame
    bcs_df = pd.DataFrame(
        {algo: [bcs[abbr][algo] for abbr, _ in BEHAVIORAL_GOALS] for algo in algos},
        index=[abbr for abbr, _ in BEHAVIORAL_GOALS]
    )
    bcs_df.index.name = 'Behaviour'

    return bcs_df, norm_scores

def export_results(bcs_df, norm_scores):
    """Export BCS table and visualizations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save BCS table
    out_fp = os.path.join(OUTPUT_DIR, 'bcs_stats_from_csv.csv')
    bcs_df.to_csv(out_fp)
    print(f"Saved BCS stats to {out_fp}")
    print("\nBCS Scores:")
    print(bcs_df.to_string())

    # Save normalized scores table
    norm_df = pd.DataFrame(norm_scores).T
    norm_out_fp = os.path.join(OUTPUT_DIR, 'normalized_scores.csv')
    norm_df.to_csv(norm_out_fp)
    print(f"\nSaved normalized scores to {norm_out_fp}")

    # Generate radar chart
    radar_df = bcs_df.reset_index()
    radar_path = os.path.join(OUTPUT_DIR, 'bcs_radar_from_csv.png')
    save_highdef_radar(
        radar_df,
        'Behavioral Competency Score (BCS)',
        radar_path,
        dpi=300,
        figsize=(10, 10)
    )
    print(f"Saved radar chart to {radar_path}")

def main():
    print("Loading CSV data...")
    df = load_data()

    print(f"\nLoaded {len(df)} levels")
    print(f"Algorithms: {[col for col in df.columns if col not in ['level', 'max_score', 'Do_Nothing']]}")

    print("\nCalculating BCS scores...")
    bcs_df, norm_scores = calculate_bcs(df)

    print("\nExporting results...")
    export_results(bcs_df, norm_scores)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
