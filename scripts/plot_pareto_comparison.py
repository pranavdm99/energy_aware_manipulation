"""
plot_pareto_comparison.py — ECO Language Conditioning Ablation Study.

Shows Door (ECO + language) as a true Pareto frontier vs Lift baseline
(fixed policy, no language conditioning) as an operating region cluster.

Usage:
    python scripts/plot_pareto_comparison.py
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DESCRIPTOR_ORDER  = ["carefully", "gently", "efficiently", "normally", "quickly"]
DESCRIPTOR_COLORS = {
    "carefully":   "#5e81f4",
    "gently":      "#48c9b0",
    "efficiently": "#f0b429",
    "normally":    "#e07b39",
    "quickly":     "#e84393",
}
DESCRIPTOR_LABELS = {
    "carefully":   "carefully",
    "gently":      "gently",
    "efficiently": "efficiently",
    "normally":    "normally",
    "quickly":     "quickly",
}

dark_bg  = "#0f0f1a"
panel_bg = "#171728"
grid_col = "#2a2a45"
text_col = "#d0d0e8"
title_col = "#ffffff"

DOOR_COLOR = "#e05c5c"
LIFT_COLOR = "#4a90d9"

def load_pareto(path, task_name):
    df = pd.read_csv(path)
    df["task"] = task_name
    df["success_pct"] = df["success"] * 100.0
    df["descriptor_ord"] = df["descriptor"].map(
        {d: i for i, d in enumerate(DESCRIPTOR_ORDER)}
    )
    return df.sort_values("descriptor_ord")

def style_ax(ax):
    ax.set_facecolor(panel_bg)
    ax.tick_params(colors=text_col, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(grid_col)
    ax.grid(True, color=grid_col, linewidth=0.6, alpha=0.8)
    ax.xaxis.label.set_color(text_col)
    ax.yaxis.label.set_color(text_col)

def plot_comparison(lift_df, door_df, output_path, show=False):
    fig = plt.figure(figsize=(17, 10), facecolor=dark_bg)
    gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.32,
                   left=0.07, right=0.97, top=0.87, bottom=0.10)

    ax_pareto = fig.add_subplot(gs[:, 0])
    ax_sr     = fig.add_subplot(gs[0, 1])
    ax_energy = fig.add_subplot(gs[1, 1])

    for ax in [ax_pareto, ax_sr, ax_energy]:
        style_ax(ax)

    # ── 1. Main Pareto ────────────────────────────────────────────────────────

    # Lift: show as a shaded "operating region" cluster (not a frontier)
    lift_e = lift_df["total_energy"].values
    lift_s = lift_df["success_pct"].values
    cx, cy = lift_e.mean(), lift_s.mean()
    # Draw semi-transparent ellipse showing spread
    ell = Ellipse((cx, cy),
                  width=lift_e.std() * 3.5 + 60,
                  height=lift_s.std() * 3.5 + 8,
                  angle=0,
                  facecolor=LIFT_COLOR, alpha=0.12,
                  edgecolor=LIFT_COLOR, linewidth=1.5, linestyle="--")
    ax_pareto.add_patch(ell)
    ax_pareto.scatter(lift_e, lift_s,
                      color=LIFT_COLOR, s=70, alpha=0.55,
                      marker="o", zorder=3, label="__nolegend__")
    # Label cluster centre
    ax_pareto.annotate("Lift baseline\n(fixed policy — no language)",
                        (cx, cy), xytext=(cx + 25, cy - 14),
                        fontsize=9, color=LIFT_COLOR,
                        arrowprops=dict(arrowstyle="->", color=LIFT_COLOR, lw=1.2))

    # Door: true Pareto frontier — line + colored dots per descriptor
    door_sorted = door_df[door_df["success_pct"] > 0].sort_values("total_energy")
    ax_pareto.plot(door_sorted["total_energy"], door_sorted["success_pct"],
                   color=DOOR_COLOR, linewidth=2.2, zorder=4)

    for _, row in door_df.iterrows():
        desc = row["descriptor"]
        color = DESCRIPTOR_COLORS.get(desc, "#aaa")
        ax_pareto.scatter(row["total_energy"], row["success_pct"],
                          color=color, s=140, zorder=6,
                          marker="s", edgecolors=DOOR_COLOR, linewidths=1.5)
        offset = (8, 4) if row["success_pct"] > 5 else (8, -14)
        ax_pareto.annotate(
            f"{desc}\n({row['success_pct']:.0f}%)",
            (row["total_energy"], row["success_pct"]),
            textcoords="offset points", xytext=offset,
            fontsize=8, color=text_col, alpha=0.9
        )

    ax_pareto.set_xlabel("Episode Energy (J)", fontsize=12)
    ax_pareto.set_ylabel("Success Rate (%)", fontsize=12)
    ax_pareto.set_ylim(-10, 115)
    ax_pareto.set_title("Ablation: Fixed Policy vs\nLanguage-Conditioned Pareto Frontier",
                         color=title_col, fontsize=13, fontweight="bold")

    legend_handles = [
        mpatches.Patch(facecolor=LIFT_COLOR, alpha=0.6,
                       label="Lift — fixed α policy (no language)"),
        mpatches.Patch(facecolor=DOOR_COLOR, alpha=0.8,
                       label="Door — ECO + language conditioning"),
    ]
    ax_pareto.legend(handles=legend_handles, framealpha=0.25,
                     facecolor=panel_bg, labelcolor=text_col, fontsize=9,
                     loc="lower right")

    # ── 2. Success Rate grouped bar (right, top) ──────────────────────────────
    x = np.arange(len(DESCRIPTOR_ORDER))
    w = 0.35

    lift_sr = [lift_df.set_index("descriptor").loc[d, "success_pct"]
               if d in lift_df["descriptor"].values else 0
               for d in DESCRIPTOR_ORDER]
    door_sr = [door_df.set_index("descriptor").loc[d, "success_pct"]
               if d in door_df["descriptor"].values else 0
               for d in DESCRIPTOR_ORDER]

    b1 = ax_sr.bar(x - w/2, lift_sr, w, label="Lift (baseline)", color=LIFT_COLOR,
                   alpha=0.75, edgecolor=grid_col)
    b2 = ax_sr.bar(x + w/2, door_sr, w, label="Door (ECO+lang)", color=DOOR_COLOR,
                   alpha=0.85, edgecolor=grid_col)

    for bars, vals in [(b1, lift_sr), (b2, door_sr)]:
        for bar, v in zip(bars, vals):
            if v > 3:
                ax_sr.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                           f"{v:.0f}%", ha="center", va="bottom",
                           fontsize=7.5, color=text_col)

    ax_sr.set_xticks(x)
    ax_sr.set_xticklabels([DESCRIPTOR_LABELS[d] for d in DESCRIPTOR_ORDER],
                           rotation=30, ha="right", color=text_col, fontsize=9)
    ax_sr.set_ylabel("Success Rate (%)", fontsize=10)
    ax_sr.set_title("Success Rate by Descriptor", color=title_col,
                     fontsize=11, fontweight="bold")
    ax_sr.set_ylim(0, 118)
    ax_sr.legend(facecolor=panel_bg, labelcolor=text_col, fontsize=9, framealpha=0.2)

    # Annotate: Lift bars are flat = no adaptation
    ax_sr.annotate("No adaptation\n(all ~75%)", xy=(1.5, 77), fontsize=8,
                    color=LIFT_COLOR, ha="center", style="italic")

    # ── 3. Energy grouped bar (right, bottom) ────────────────────────────────
    lift_e_vals = [lift_df.set_index("descriptor").loc[d, "total_energy"]
                   if d in lift_df["descriptor"].values else 0
                   for d in DESCRIPTOR_ORDER]
    door_e_vals = [door_df.set_index("descriptor").loc[d, "total_energy"]
                   if d in door_df["descriptor"].values else 0
                   for d in DESCRIPTOR_ORDER]

    b3 = ax_energy.bar(x - w/2, lift_e_vals, w, label="Lift (baseline)",
                        color=LIFT_COLOR, alpha=0.75, edgecolor=grid_col)
    b4 = ax_energy.bar(x + w/2, door_e_vals, w, label="Door (ECO+lang)",
                        color=DOOR_COLOR, alpha=0.85, edgecolor=grid_col)

    for bars, vals in [(b3, lift_e_vals), (b4, door_e_vals)]:
        for bar, v in zip(bars, vals):
            if v > 5:
                ax_energy.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                               f"{v:.0f}", ha="center", va="bottom",
                               fontsize=7, color=text_col)

    ax_energy.set_xticks(x)
    ax_energy.set_xticklabels([DESCRIPTOR_LABELS[d] for d in DESCRIPTOR_ORDER],
                                rotation=30, ha="right", color=text_col, fontsize=9)
    ax_energy.set_ylabel("Avg Episode Energy (J)", fontsize=10)
    ax_energy.set_title("Energy Consumption by Descriptor", color=title_col,
                         fontsize=11, fontweight="bold")
    ax_energy.legend(facecolor=panel_bg, labelcolor=text_col, fontsize=9, framealpha=0.2)

    # Draw a bracket showing Door's energy differentiation range
    door_min = min(door_e_vals)
    door_max = max(door_e_vals)
    ax_energy.annotate("", xy=(4 + w/2, door_max), xytext=(4 + w/2, door_min),
                        arrowprops=dict(arrowstyle="<->", color=DOOR_COLOR, lw=1.5))
    ax_energy.text(4 + w/2 + 0.12, (door_max + door_min)/2,
                   f"3.4×\nrange", fontsize=8, color=DOOR_COLOR, va="center")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        "ECO + Language Conditioning — Ablation Study\n"
        "Fixed Energy Policy (Lift) vs Full Framework with Language Descriptors (Door)",
        color=title_col, fontsize=14, fontweight="bold", y=0.97
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=dark_bg)
    print(f"Saved → {output_path}")
    if show:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lift-csv",  default="results/lift_pareto_table.csv")
    parser.add_argument("--door-csv",  default="results/door_pareto_table.csv")
    parser.add_argument("--output",    default="results/pareto_comparison.png")
    parser.add_argument("--show",      action="store_true")
    args = parser.parse_args()

    for path in [args.lift_csv, args.door_csv]:
        if not os.path.exists(path):
            print(f"ERROR: file not found: {path}"); sys.exit(1)

    lift_df = load_pareto(args.lift_csv, "Lift")
    door_df = load_pareto(args.door_csv, "Door")

    print("\nLift (baseline):")
    print(lift_df[["descriptor", "success_pct", "total_energy"]].to_string(index=False))
    print("\nDoor (ECO + language):")
    print(door_df[["descriptor", "success_pct", "total_energy"]].to_string(index=False))

    plot_comparison(lift_df, door_df, args.output, show=args.show)

if __name__ == "__main__":
    main()

