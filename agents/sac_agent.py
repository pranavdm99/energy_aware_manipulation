"""
SAC Agent — Wrapper around Stable-Baselines3 SAC for energy-aware training.

Handles model creation, training with callbacks, checkpointing,
and evaluation with energy metric collection.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

from utils.logging_utils import EnergyLoggingCallback


def create_sac_agent(
    env,
    learning_rate: float = 3e-4,
    buffer_size: int = 1_000_000,
    batch_size: int = 256,
    tau: float = 0.005,
    gamma: float = 0.99,
    train_freq: int = 1,
    gradient_steps: int = 1,
    learning_starts: int = 10_000,
    policy_kwargs: Optional[Dict] = None,
    seed: int = 42,
    device: str = "auto",
    verbose: int = 1,
) -> SAC:
    """Create a configured SAC agent.

    Args:
        env: Gymnasium-compatible environment.
        learning_rate: Learning rate for actor and critic.
        buffer_size: Replay buffer capacity.
        batch_size: Mini-batch size for updates.
        tau: Soft target update coefficient.
        gamma: Discount factor.
        train_freq: Update frequency (steps).
        gradient_steps: Gradient steps per update.
        learning_starts: Steps before training begins.
        policy_kwargs: Extra kwargs for MlpPolicy (e.g., net_arch).
        seed: Random seed.
        device: 'auto', 'cuda', or 'cpu'.
        verbose: Verbosity level.

    Returns:
        Configured SAC model (untrained).
    """
    if policy_kwargs is None:
        policy_kwargs = {"net_arch": [256, 256]}

    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        learning_starts=learning_starts,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        verbose=verbose,
    )

    return model


def create_sac_from_config(env, config: Dict[str, Any]) -> SAC:
    """Create SAC agent from a configuration dictionary.

    Args:
        env: Gymnasium-compatible environment.
        config: Full config dict with 'training' section.

    Returns:
        Configured SAC model.
    """
    train_cfg = config.get("training", {})
    return create_sac_agent(
        env=env,
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        buffer_size=train_cfg.get("buffer_size", 1_000_000),
        batch_size=train_cfg.get("batch_size", 512),
        tau=train_cfg.get("tau", 0.005),
        gamma=train_cfg.get("gamma", 0.99),
        train_freq=train_cfg.get("train_freq", 1),
        gradient_steps=train_cfg.get("gradient_steps", 1),
        learning_starts=train_cfg.get("learning_starts", 1000),
        policy_kwargs=train_cfg.get("policy_kwargs", {"net_arch": [256, 256]}),
        seed=train_cfg.get("seed", 42),
    )


def train_agent(
    model: SAC,
    total_timesteps: int = 500_000,
    eval_env=None,
    checkpoint_dir: str = "checkpoints",
    log_freq: int = 1000,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 20,
    save_freq: int = 50_000,
    verbose: int = 1,
) -> SAC:
    """Train the SAC agent with energy logging and checkpointing.

    Args:
        model: Pre-configured SAC model.
        total_timesteps: Total training steps.
        eval_env: Optional separate env for evaluation.
        checkpoint_dir: Directory for saving checkpoints.
        log_freq: WandB logging frequency.
        eval_freq: Evaluation frequency.
        n_eval_episodes: Episodes per evaluation.
        save_freq: Checkpoint save frequency.
        verbose: Verbosity.

    Returns:
        Trained SAC model.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Build callbacks ---
    callbacks = []

    # WandB callback for SB3 native metrics (actor_loss, critic_loss, etc.)
    try:
        from wandb.integration.sb3 import WandbCallback
        import wandb as _wandb
        if _wandb.run is not None:
            wandb_cb = WandbCallback(
                model_save_path=None,  # we handle saving separately
                verbose=verbose,
            )
            callbacks.append(wandb_cb)
    except ImportError:
        pass  # WandB integration not available

    # Energy logging to WandB
    energy_cb = EnergyLoggingCallback(log_freq=log_freq, verbose=verbose)
    callbacks.append(energy_cb)

    # Periodic checkpoints
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix="sac_energy",
        verbose=verbose,
    )
    callbacks.append(checkpoint_cb)

    # Evaluation callback (if eval env provided)
    if eval_env is not None:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(checkpoint_dir, "best_model"),
            log_path=os.path.join(checkpoint_dir, "eval_logs"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=verbose,
        )
        callbacks.append(eval_cb)

    callback = CallbackList(callbacks)

    # --- Train ---
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # --- Save final model ---
    final_path = os.path.join(checkpoint_dir, "sac_final")
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    return model


def evaluate_agent(
    model: SAC,
    env,
    n_episodes: int = 100,
    deterministic: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a trained agent and collect energy metrics.

    Args:
        model: Trained SAC model.
        env: Evaluation environment (with EnergyAwareWrapper).
        n_episodes: Number of evaluation episodes.
        deterministic: Use deterministic actions.
        verbose: Print progress.

    Returns:
        Dict with aggregated evaluation metrics.
    """
    results = {
        "successes": [],
        "total_energies": [],
        "peak_torques": [],
        "episode_lengths": [],
        "total_rewards": [],
    }

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        # Collect episode summary
        ep_summary = info.get("energy", {}).get("episode_summary", {})
        results["total_energies"].append(ep_summary.get("total_energy", 0.0))
        results["peak_torques"].append(ep_summary.get("peak_torque", 0.0))
        results["episode_lengths"].append(ep_summary.get("episode_length", 0))
        results["total_rewards"].append(ep_reward)
        results["successes"].append(float(info.get("is_success", False)))

        if verbose and (ep + 1) % 10 == 0:
            print(
                f"Episode {ep + 1}/{n_episodes} | "
                f"Success rate: {np.mean(results['successes']):.2%} | "
                f"Avg energy: {np.mean(results['total_energies']):.2f}"
            )

    # Aggregate
    summary = {
        "success_rate": np.mean(results["successes"]),
        "mean_total_energy": np.mean(results["total_energies"]),
        "std_total_energy": np.std(results["total_energies"]),
        "mean_peak_torque": np.mean(results["peak_torques"]),
        "std_peak_torque": np.std(results["peak_torques"]),
        "mean_episode_length": np.mean(results["episode_lengths"]),
        "mean_total_reward": np.mean(results["total_rewards"]),
        "n_episodes": n_episodes,
    }

    return summary
