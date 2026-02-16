"""
ablation_sweep.py — Automated ablation runner across energy weights and seeds.

Usage:
    python scripts/ablation_sweep.py
    python scripts/ablation_sweep.py --config configs/ablation_sweep.yaml
"""

import argparse
import itertools
import os
import subprocess
import sys
import yaml
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation sweep")
    parser.add_argument("--config", type=str, default="configs/ablation_sweep.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep = sweep_config["sweep"]
    values = sweep["values"]
    seeds = sweep["seeds"]
    tasks = sweep["tasks"]
    total_timesteps = sweep.get("total_timesteps", 500_000)

    # Generate all combinations
    combinations = list(itertools.product(tasks, values, seeds))
    print(f"Total runs: {len(combinations)}")
    print(f"Tasks: {tasks}")
    print(f"Energy weights: {values}")
    print(f"Seeds: {seeds}")
    print(f"Timesteps per run: {total_timesteps}")
    print()

    for i, (task, alpha, seed) in enumerate(combinations):
        cmd = [
            sys.executable, "scripts/train.py",
            "--task", task,
            "--energy-weight", str(alpha),
            "--total-timesteps", str(total_timesteps),
            "--seed", str(seed),
            "--n-envs", "8",
            "--horizon", "400",
            "--learning-starts", "10000",
            "--wandb-mode", "online",
            "--checkpoint-dir", f"checkpoints/sweep",
            "--batch-size", "512",
            "--gradient-steps", "1",
            "--train-freq", "1",
        ]

        run_name = f"{task}_alpha{alpha}_seed{seed}_{int(time.time())}"
        print(f"[{i + 1}/{len(combinations)}] {run_name}")
        print(f"  Command: {' '.join(cmd)}")

        if not args.dry_run:
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                print(f"  ERROR: Run failed with code {result.returncode}")
            else:
                print(f"  DONE")
        print()

    print("Ablation sweep complete!")


if __name__ == "__main__":
    main()
