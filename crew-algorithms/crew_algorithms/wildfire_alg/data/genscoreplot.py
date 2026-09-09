import matplotlib.pyplot as plt
import numpy as np

# ====== Data ======
levels = ["Cut_Trees_Sparse_large", "Rescue_Civilians_Known_Location_large", "Transport_Firefighters_large"]

methods = ["Human", "Simple", "Critic", "No-Critic", "CAMON", "COELA", "Embodied", "HMAS-2", "Do Nothing", "Max"]


means_data = {
    "Cut_Trees_Sparse_large": [74.60, 68.20, 75.00, 74.40, 72.00, 57.67, 50.33, 56.33, 0.00],
    "Rescue_Civilians_Known_Location_large": [6.80, 5.60, 8.20, 6.00, 4.00, 0.00, 4.33, 4.00, 0.00],
    "Transport_Firefighters_large": [5.00, 10.00, 10.00, 10.00, 10.00, 8.33, 9.33, 8.33, 0.00],
}

std_data = {
    "Cut_Trees_Sparse_large": [0.55, 4.38, 0.00, 1.34, 4.00, 25.00, 9.00, 13.00, 0.00],
    "Rescue_Civilians_Known_Location_large": [1.10, 0.89, 0.84, 1.22, 5.00, 0.00, 3.00, 2.00, 0.00],
    "Transport_Firefighters_large": [5.00, 0.00, 0.00, 0.00, 0.00, 5.00, 2.00, 5.00, 0.00],
}

# ====== Colors ======
colors = ["#1f77b4"] * 4 + ["#ff7f0e"] * 4 + ["#7f7f7f", "#a6a6a6"]  # last gray = Max

# ====== Plotting ======
for level in levels:
    means = np.array(means_data[level])
    stds = np.array(std_data[level])
    
    # Compute max score for this level
    level_max = means.max()
    
    # Append the max bar
    means = np.append(means, level_max)
    stds = np.append(stds, 0.0)
    
    # Clip error bars so they stay within [0, level_max]
    lower_err = np.minimum(stds, means)              # ensure mean - std ≥ 0
    upper_err = np.minimum(stds, level_max - means)  # ensure mean + std ≤ level_max
    yerr = [lower_err, upper_err]
    
    x = np.arange(len(methods))
    
    plt.figure(figsize=(8, 4))
    plt.bar(x, means, color=colors, yerr=yerr, capsize=5, ecolor='black')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.ylabel("Score (mean ± std)")
    plt.title(level.replace("_", " "))
    plt.ylim(0, level_max * 1.1)
    plt.tight_layout()
    plt.show()
