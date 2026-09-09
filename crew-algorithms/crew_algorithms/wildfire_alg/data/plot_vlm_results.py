"""
Plot VLM Comparison Results

Reads vlm_results.json and generates visualization plots comparing
Perception Module vs VLM performance.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def load_results(json_path: str) -> dict:
    """Load results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)


def compute_statistics(results: dict) -> dict:
    """Compute statistics including standard deviations"""
    stats = {}

    for method in ["perception_module", "vlm"]:
        samples = results[method]["samples"]

        # Extract arrays
        accuracies = np.array([s["accuracy"] for s in samples])
        fire_scores = np.array([s["fire_score"] for s in samples])
        civilian_scores = np.array([s["civilian_score"] for s in samples])
        times = np.array([s["time"] for s in samples])

        stats[method] = {
            "avg_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "avg_fire_score": float(np.mean(fire_scores)),
            "std_fire_score": float(np.std(fire_scores)),
            "avg_civilian_score": float(np.mean(civilian_scores)),
            "std_civilian_score": float(np.std(civilian_scores)),
            "avg_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "total_cost": results[method]["total_cost"],
            "total_tokens": results[method]["total_tokens"],
            "num_samples": len(samples)
        }

    return stats


def compute_wins(results: dict) -> tuple:
    """Compute win/tie/loss counts based on overall accuracy"""
    perception_samples = results['perception_module']['samples']
    vlm_samples = results['vlm']['samples']

    # Count wins/ties/losses
    perception_win_count = 0
    vlm_win_count = 0
    tie_count = 0

    for i in range(len(perception_samples)):
        p_acc = perception_samples[i]['accuracy']
        v_acc = vlm_samples[i]['accuracy']

        # Determine winner (using a small epsilon for ties)
        if abs(p_acc - v_acc) < 1.0:  # Within 1 point = tie
            tie_count += 1
        elif p_acc > v_acc:
            perception_win_count += 1
        else:
            vlm_win_count += 1

    return perception_win_count, vlm_win_count, tie_count


def generate_plots(stats: dict, results: dict, output_path: Path):
    """Generate comparison plots in a single image"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VLM vs Perception Module Comparison', fontsize=16, fontweight='bold')

    # Colors
    perception_color = '#2E86AB'
    vlm_color = '#A23B72'

    # Data
    perception_data = stats['perception_module']
    vlm_data = stats['vlm']

    # Plot 1: Accuracy Scores
    ax1 = axes[0, 0]
    metrics = ['Overall\nAccuracy', 'Fire\nDetection', 'Civilian\nDetection']
    perception_scores = [
        perception_data['avg_accuracy'],
        perception_data['avg_fire_score'],
        perception_data['avg_civilian_score']
    ]
    vlm_scores = [
        vlm_data['avg_accuracy'],
        vlm_data['avg_fire_score'],
        vlm_data['avg_civilian_score']
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax1.bar(x - width/2, perception_scores, width,
            label='Perception Module', color=perception_color, alpha=0.8)
    ax1.bar(x + width/2, vlm_scores, width,
            label='VLM', color=vlm_color, alpha=0.8)

    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Accuracy Scores', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0, 10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Plot 2: Average Time per Sample
    ax2 = axes[0, 1]
    methods = ['Perception\nModule', 'VLM']
    times = [perception_data['avg_time'], vlm_data['avg_time']]
    colors = [perception_color, vlm_color]

    bars = ax2.bar(methods, times, color=colors, alpha=0.8)
    ax2.set_ylabel('Time (seconds)', fontsize=11)
    ax2.set_title('Average Time per Sample', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{time:.3f}s', ha='center', va='bottom', fontsize=10)

    # Plot 3: Total Cost
    ax3 = axes[1, 0]
    costs = [perception_data['total_cost'], vlm_data['total_cost']]
    bars = ax3.bar(methods, costs, color=colors, alpha=0.8)
    ax3.set_ylabel('Cost ($)', fontsize=11)
    ax3.set_title('Total API Cost', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'${cost:.4f}', ha='center', va='bottom', fontsize=10)

    # Plot 4: Win/Tie/Loss (as percentage)
    ax4 = axes[1, 1]
    perception_win_count, vlm_win_count, tie_count = compute_wins(results)

    total = perception_win_count + vlm_win_count + tie_count
    perception_win_pct = (perception_win_count / total) * 100
    vlm_win_pct = (vlm_win_count / total) * 100
    tie_pct = (tie_count / total) * 100

    categories = ['Perception\nModule Wins', 'VLM Wins', 'Ties']
    percentages = [perception_win_pct, vlm_win_pct, tie_pct]
    win_colors = [perception_color, vlm_color, '#F18F01']

    bars = ax4.bar(categories, percentages, color=win_colors, alpha=0.8)
    ax4.set_ylabel('Percentage (%)', fontsize=11)
    ax4.set_title('Win/Tie Comparison (Overall Accuracy)', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 105)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    """Main entry point"""
    # Default path (can be changed)
    json_path = "vlm_results.json"

    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    json_path = Path(json_path)

    if not json_path.exists():
        print(f"Error: {json_path} not found!")
        print(f"Usage: python plot_vlm_results.py [path/to/vlm_results.json]")
        sys.exit(1)

    # Load results
    print(f"Loading results from: {json_path}")
    results = load_results(json_path)

    # Compute statistics
    print("Computing statistics...")
    stats = compute_statistics(results)

    # Generate comparison plots (all in one image)
    output_path = json_path.parent / f"{json_path.stem}_comparison.png"
    print("Generating comparison plots...")
    generate_plots(stats, results, output_path)

    # Print summary
    print("\n" + "="*60)
    print("STATISTICS SUMMARY")
    print("="*60)

    for method in ["perception_module", "vlm"]:
        method_name = "Perception Module" if method == "perception_module" else "VLM"
        data = stats[method]
        print(f"\n{method_name}:")
        print(f"  Samples: {data['num_samples']}")
        print(f"  Avg Accuracy: {data['avg_accuracy']:.2f} ± {data['std_accuracy']:.2f}")
        print(f"  Avg Fire Score: {data['avg_fire_score']:.2f} ± {data['std_fire_score']:.2f}")
        print(f"  Avg Civilian Score: {data['avg_civilian_score']:.2f} ± {data['std_civilian_score']:.2f}")
        print(f"  Avg Time: {data['avg_time']:.3f}s ± {data['std_time']:.3f}s")
        print(f"  Total Cost: ${data['total_cost']:.4f}")
        print(f"  Total Tokens: {data['total_tokens']}")


if __name__ == "__main__":
    main()
