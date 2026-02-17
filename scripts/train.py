"""
train.py — Main training entry point for energy-aware manipulation.

Usage:
    python scripts/train.py --task Lift --energy-weight 0.0 --total-timesteps 500000
    python scripts/train.py --task Lift --energy-weight 0.0 --n-envs 8 --gradient-steps 4
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --task Lift --energy-weight 0.1 --language-conditioned --descriptor "gently"
"""

import argparse
import os
import sys
import yaml
import time

# Headless rendering for parallel envs
os.environ.setdefault("MUJOCO_GL", "osmesa")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from envs.env_factory import make_env, make_env_from_config
from agents.sac_agent import create_sac_agent, create_sac_from_config, train_agent
from utils.logging_utils import init_wandb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train energy-aware manipulation agent"
    )

    # Config file (overrides individual args)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")

    # Environment
    parser.add_argument("--task", type=str, default="Lift",
                        help="Robosuite task (Lift, PickPlace, Door, NutAssemblySingle)")
    parser.add_argument("--horizon", type=int, default=500,
                        help="Max steps per episode")

    # Energy
    parser.add_argument("--energy-weight", type=float, default=0.0,
                        help="Energy penalty weight (alpha)")
    parser.add_argument("--include-energy-obs", action="store_true",
                        help="Include torque features in observations")
    parser.add_argument("--include-energy-in-obs", action="store_true", help="Include energy metrics in observation")
    
    # Language arguments
    parser.add_argument("--language-conditioned", action="store_true", help="Enable language conditioning")
    parser.add_argument("--descriptor", type=str, default="normally", help="Task descriptor (e.g., 'gently', 'quickly')")
    parser.add_argument("--randomize-descriptor", action="store_true", help="Randomly sample descriptors at each reset")
    parser.add_argument("--language-model", type=str, default="all-MiniLM-L6-v2", help="Sentence-BERT model name")

    # Training arguments
    parser.add_argument("--total-timesteps", type=int, default=500000,
                        help="Total training timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Mini-batch size for gradient updates")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments (1=DummyVecEnv, >1=SubprocVecEnv)")
    parser.add_argument("--gradient-steps", type=int, default=1,
                        help="Gradient updates per env step (increase for GPU utilization)")
    parser.add_argument("--train-freq", type=int, default=1,
                        help="Env steps between gradient updates")
    parser.add_argument("--learning-starts", type=int, default=1000,
                        help="Random exploration steps before training begins")
    parser.add_argument("--seed", type=int, default=42)

    # Logging
    parser.add_argument("--wandb-project", type=str,
                        default="energy-aware-manipulation")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--save-freq", type=int, default=50_000)

    # Output
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    return parser.parse_args()


def _make_env_fn(config, rank, seed):
    """Create a callable that returns an environment (for SubprocVecEnv)."""
    def _init():
        env = make_env_from_config(config)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_vec_env(config, n_envs, seed):
    """Create a vectorized environment (parallel or sequential).

    Args:
        config: Environment config dict.
        n_envs: Number of parallel environments.
        seed: Base random seed.

    Returns:
        VecEnv instance (SubprocVecEnv if n_envs > 1, else DummyVecEnv).
    """
    env_fns = [_make_env_fn(config, i, seed) for i in range(n_envs)]

    if n_envs > 1:
        print(f"Creating {n_envs} parallel environments (SubprocVecEnv)...")
        return SubprocVecEnv(env_fns, start_method="forkserver")
    else:
        print("Creating single environment (DummyVecEnv)...")
        return DummyVecEnv(env_fns)


def main():
    args = parse_args()

    # --- Load config or build from args ---
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        # Only override with CLI when not default (so config wandb_mode is preserved when user doesn't pass --wandb-mode)
        if args.wandb_mode != "online":
            config.setdefault("logging", {})["wandb_mode"] = args.wandb_mode
    else:
        config = {
            "environment": {
                "task": args.task,
                "robots": "Panda",
                "controller": "OSC_POSE",
                "horizon": args.horizon,
                "reward_shaping": True,
            },
            "training": {
                "total_timesteps": args.total_timesteps,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "n_envs": args.n_envs,
                "gradient_steps": args.gradient_steps,
                "train_freq": args.train_freq,
                "learning_starts": args.learning_starts,
                "seed": args.seed,
                "learning_starts": args.learning_starts,
                "seed": args.seed,
            },
            "energy": {
                "weight": args.energy_weight,
                "normalize_by_dof": True,
                "include_in_obs": args.include_energy_in_obs,
            },
            "language": {
                "enabled": args.language_conditioned,
                "descriptor": args.descriptor,
                "model": args.language_model,
                "randomize_descriptor": args.randomize_descriptor,
            },
            "logging": {
                "wandb_project": args.wandb_project,
                "wandb_mode": args.wandb_mode,
                "log_interval": args.log_interval,
                "eval_freq": args.eval_freq,
                "save_freq": args.save_freq,
            },
        }

    # --- Generate run name ---
    energy_w = config["energy"]["weight"]
    task = config["environment"]["task"]
    seed = config["training"]["seed"]
    n_envs = config["training"].get("n_envs", 1)
    lang_tag = ""
    if config["language"]["enabled"]:
        lang_tag = f"_lang-{config['language']['descriptor']}"
    run_name = f"{task}_alpha{energy_w}{lang_tag}_seed{seed}_{int(time.time())}"

    # --- Init WandB ---
    # Use explicit CLI/config mode if set; otherwise let init_wandb read WANDB_MODE env var
    logging_cfg = config.get("logging", {})
    wandb_mode = logging_cfg.get("wandb_mode", None)  # None = respect WANDB_MODE env var
    init_wandb(
        project=logging_cfg.get("wandb_project", "energy-aware-manipulation"),
        config=config,
        run_name=run_name,
        mode=wandb_mode,
    )

    # --- Create environments ---
    print(f"\n{'='*60}")
    print(f"Training: {run_name}")
    print(f"Task: {task} | Energy weight: {energy_w}")
    print(f"Parallel envs: {n_envs} | Gradient steps: {config['training'].get('gradient_steps', 1)}")
    print(f"Batch size: {config['training']['batch_size']} | Total timesteps: {config['training']['total_timesteps']}")
    print(f"{'='*60}\n")

    env = create_vec_env(config, n_envs, seed)

    # Separate eval env (single env, no energy penalty for fair success eval)
    eval_config = config.copy()
    eval_config["energy"] = {**config["energy"], "weight": 0.0}
    eval_env = create_vec_env(eval_config, n_envs=1, seed=seed + 100)

    # --- Create and train agent ---
    model = create_sac_from_config(env, config)

    checkpoint_dir = os.path.join(
        config.get("paths", {}).get("checkpoint_dir", args.checkpoint_dir),
        run_name,
    )

    model = train_agent(
        model=model,
        total_timesteps=config["training"]["total_timesteps"],
        eval_env=eval_env,
        checkpoint_dir=checkpoint_dir,
        log_freq=config.get("logging", {}).get("log_interval", 1000),
        eval_freq=config.get("logging", {}).get("eval_freq", 10_000),
        save_freq=config.get("logging", {}).get("save_freq", 50_000),
    )

    print(f"\nTraining complete! Model saved to {checkpoint_dir}")

    # --- Quick evaluation ---
    from agents.sac_agent import evaluate_agent

    # Use single env for final eval
    single_eval_env = make_env_from_config(eval_config)
    print("\nRunning final evaluation (20 episodes)...")
    results = evaluate_agent(model, single_eval_env, n_episodes=20)
    print(f"\nFinal Results:")
    print(f"  Success rate:   {results['success_rate']:.2%}")
    print(f"  Mean energy:    {results['mean_total_energy']:.2f}")
    print(f"  Mean peak τ:    {results['mean_peak_torque']:.2f}")
    print(f"  Mean ep length: {results['mean_episode_length']:.0f}")

    env.close()
    eval_env.close()
    single_eval_env.close()


if __name__ == "__main__":
    main()
