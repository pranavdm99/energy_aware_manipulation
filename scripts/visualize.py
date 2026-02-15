"""
visualize.py — Publication-quality plotting for energy-aware manipulation results.

Generates: Pareto front (success vs energy), torque heatmaps,
smoothness comparisons, and energy profiles.

Usage:
    python scripts/visualize.py --results-dir results/
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Use non-interactive backend for Docker/headless
matplotlib.use("Agg")

# Publication-quality defaults
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "lines.linewidth": 2,
    "lines.markersize": 8,
})


def load_results(results_dir):
    """Load all JSON result files from a directory."""
    results = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname), "r") as f:
                data = json.load(f)
            key = fname.replace(".json", "")
            results[key] = data
    return results


def plot_pareto_front(results, output_dir):
    """Plot success rate vs. total energy Pareto front."""
    fig, ax = plt.subplots()

    alphas = []
    success_rates = []
    mean_energies = []

    for name, data in results.items():
        sr = np.mean([r["success"] for r in data])
        energy = np.mean([r["total_energy"] for r in data])
        success_rates.append(sr)
        mean_energies.append(energy)

        # Extract alpha from name
        if "alpha" in name:
            alpha = float(name.split("alpha")[1].split("_")[0])
        else:
            alpha = 0.0
        alphas.append(alpha)

    scatter = ax.scatter(
        mean_energies, success_rates,
        c=alphas, cmap="viridis", s=100, edgecolors="black", zorder=5
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Energy weight (α)")

    ax.set_xlabel("Mean Total Energy (J·s)")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate vs. Energy Consumption")
    ax.grid(True, alpha=0.3)

    # Annotate points
    for i, name in enumerate(results.keys()):
        ax.annotate(
            f"α={alphas[i]}",
            (mean_energies[i], success_rates[i]),
            textcoords="offset points", xytext=(5, 5),
            fontsize=9, alpha=0.7,
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "pareto_front.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_torque_distribution(results, output_dir):
    """Plot per-joint torque distribution comparison across alpha values."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5),
                             sharey=True)
    if len(results) == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, results.items()):
        all_torque_stds = [r.get("torque_std_per_joint", []) for r in data]
        valid = [t for t in all_torque_stds if len(t) > 0]

        if valid:
            torque_std = np.mean(valid, axis=0)
            joints = list(range(len(torque_std)))
            ax.bar(joints, torque_std, color=sns.color_palette("viridis", len(joints)),
                   edgecolor="black")
            ax.set_xlabel("Joint Index")
            ax.set_title(name, fontsize=10)

    axes[0].set_ylabel("Torque Std Dev")
    fig.suptitle("Per-Joint Torque Distribution", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "torque_distribution.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_energy_comparison(results, output_dir):
    """Box plot comparing energy distributions across alpha values."""
    fig, ax = plt.subplots()

    labels = []
    energy_data = []

    for name, data in results.items():
        energies = [r["total_energy"] for r in data]
        energy_data.append(energies)
        if "alpha" in name:
            alpha = name.split("alpha")[1].split("_")[0]
            labels.append(f"α={alpha}")
        else:
            labels.append(name)

    bp = ax.boxplot(energy_data, labels=labels, patch_artist=True)
    colors = sns.color_palette("viridis", len(energy_data))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_xlabel("Energy Weight Configuration")
    ax.set_ylabel("Total Episode Energy")
    ax.set_title("Energy Distribution by α")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "energy_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots")
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--output-dir", type=str, default="results/plots/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.results_dir):
        print(f"Results directory {args.results_dir} not found.")
        print("Run evaluation first with scripts/evaluate.py")
        return

    results = load_results(args.results_dir)
    if not results:
        print("No JSON result files found.")
        return

    print(f"Loaded {len(results)} result sets")

    plot_pareto_front(results, args.output_dir)
    plot_torque_distribution(results, args.output_dir)
    plot_energy_comparison(results, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
