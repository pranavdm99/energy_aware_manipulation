
"""
evaluate_language.py — Evaluate semantic modulation of energy profiles.

Iterates through a set of task descriptors (e.g. "gently", "quickly")
and records performance metrics to verify that the policy adapts its behavior.

Usage:
    python scripts/evaluate_language.py --checkpoint checkpoints/sac_lang.zip
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.env_factory import make_env
from utils.metrics import compute_jerk

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate language-conditioned agent")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--task", type=str, default="Lift")
    parser.add_argument("--n-episodes", type=int, default=20, help="Episodes per descriptor")
    parser.add_argument("--descriptors", nargs="+", default=["gently", "normally", "quickly"],
                        help="List of descriptors to evaluate")
    parser.add_argument("--output", type=str, default="results/language_eval.csv")
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()

def set_descriptor(env, descriptor):
    """Recursively find LanguageConditionedWrapper and set descriptor."""
    current = env
    while hasattr(current, "env"):
        if hasattr(current, "set_descriptor"):
            current.set_descriptor(descriptor)
            return True
        current = current.env
    
    # Check the base env as well
    if hasattr(current, "set_descriptor"):
        current.set_descriptor(descriptor)
        return True
        
    return False

def run_evaluation(model, env, n_episodes):
    metrics = {
        "success": [],
        "total_energy": [],
        "peak_torque": [],
        "episode_length": [],
        "jerk": []
    }
    
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_energy = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        ep_summary = info.get("energy", {}).get("episode_summary", {})
        
        metrics["success"].append(float(info.get("is_success", False)))
        metrics["total_energy"].append(ep_summary.get("total_energy", 0.0))
        metrics["peak_torque"].append(ep_summary.get("peak_torque", 0.0))
        metrics["episode_length"].append(ep_summary.get("episode_length", 0))
        # Jerk is not currently in episode_summary, but logged in rolling buffer.
        # Ideally we'd compute it or extract it.
        # For now, let's use episode length as a proxy for speed, and energy as proxy for smoothness.
        
    return {k: np.mean(v) for k, v in metrics.items()}

def main():
    args = parse_args()
    
    # Create env with language conditioning enabled
    env = make_env(
        task=args.task,
        language_conditioned=True,
        descriptor="normally", # Initial
        render=args.render
    )
    
    print(f"Loading model from {args.checkpoint}")
    model = SAC.load(args.checkpoint, env=env)
    
    results = []
    
    print(f"\nEvaluating Semantic Modulation ({args.n_episodes} episodes/descriptor)")
    print(f"{'='*80}")
    print(f"{'Descriptor':<15} | {'Success':<8} | {'Energy':<10} | {'Peak Torque':<12} | {'Length':<8}")
    print(f"{'-'*80}")
    
    for desc in args.descriptors:
        # Update descriptor
        if not set_descriptor(env, desc):
            print(f"Error: Could not set descriptor to '{desc}'")
            continue
            
        # Run eval
        stats = run_evaluation(model, env, args.n_episodes)
        
        print(f"{desc:<15} | {stats['success']:.1%}   | {stats['total_energy']:<10.2f} | {stats['peak_torque']:<12.2f} | {stats['episode_length']:<8.1f}")
        
        stats["descriptor"] = desc
        results.append(stats)
        
    print(f"{'='*80}\n")
    
    # Save results
    if args.output:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

    env.close()

if __name__ == "__main__":
    main()
