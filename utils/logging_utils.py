"""
WandB logging callback for Stable-Baselines3.

Logs energy metrics, reward components, and success rates during training.
"""

import os
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Any, Optional


class EnergyLoggingCallback(BaseCallback):
    """SB3 callback that logs energy-aware metrics to WandB.

    Tracks per-episode: task reward, energy penalty, total energy,
    peak torque, mean power, success rate, and episode length.

    Args:
        log_freq: Log metrics every N steps.
        verbose: Verbosity level.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

        # Rolling buffers for episode-level metrics
        self._episode_rewards_task = []
        self._episode_rewards_energy = []
        self._episode_energies = []
        self._episode_peak_torques = []
        self._episode_jerks = []
        self._episode_contact_forces = []
        self._episode_lengths = []
        self._episode_successes = []
        self._episode_reached = []
        self._episode_grasped = []

        # Per-env episode reward accumulators
        self._step_reward_task_per_env = []
        self._step_reward_energy_per_env = []

    def _on_step(self) -> bool:
        """Called at each environment step."""
        raw_infos = self.locals.get("infos", self.locals.get("info"))
        if isinstance(raw_infos, dict):
            raw_infos = [raw_infos]
        infos = list(raw_infos) if isinstance(raw_infos, (list, tuple)) else []

        # Lazy-init per-env accumulators
        n_envs = len(infos)
        if n_envs > 0 and len(self._step_reward_task_per_env) != n_envs:
            self._step_reward_task_per_env = [0.0] * n_envs
            self._step_reward_energy_per_env = [0.0] * n_envs

        for i, info in enumerate(infos):
            energy_info = info.get("energy", {})
            if energy_info and i < len(self._step_reward_task_per_env):
                self._step_reward_task_per_env[i] += energy_info.get("reward_task", 0.0)
                self._step_reward_energy_per_env[i] += energy_info.get(
                    "reward_energy_penalty", 0.0
                )

            ep_summary = energy_info.get("episode_summary")

            if ep_summary is None:
                terminal_info = info.get("terminal_info", {})
                if isinstance(terminal_info, dict):
                    terminal_energy = terminal_info.get("energy", {})
                    ep_summary = terminal_energy.get("episode_summary")
                    if ep_summary is not None:
                        for key in ["is_success", "is_reached", "is_grasped"]:
                             if key not in info:
                                info[key] = terminal_info.get(key, False)

            if ep_summary is not None:
                self._episode_rewards_task.append(self._step_reward_task_per_env[i])
                self._episode_rewards_energy.append(self._step_reward_energy_per_env[i])
                self._episode_energies.append(ep_summary.get("total_energy", 0.0))
                self._episode_peak_torques.append(ep_summary.get("peak_torque", 0.0))
                self._episode_jerks.append(ep_summary.get("mean_jerk", 0.0))
                self._episode_contact_forces.append(ep_summary.get("max_contact_force", 0.0))
                self._episode_lengths.append(ep_summary.get("episode_length", 0))
                self._episode_successes.append(float(info.get("is_success", False)))
                self._episode_reached.append(float(info.get("is_reached", False)))
                self._episode_grasped.append(float(info.get("is_grasped", False)))
                
                self._step_reward_task_per_env[i] = 0.0
                self._step_reward_energy_per_env[i] = 0.0

        if self.num_timesteps % self.log_freq == 0 and len(self._episode_energies) > 0:
            self._log_metrics()

        return True

    def _log_metrics(self):
        """Log aggregated metrics to WandB."""
        n = len(self._episode_energies)
        if n == 0:
            return

        window = 100
        metrics = {
            "energy/mean_total_energy": np.mean(self._episode_energies[-window:]),
            "energy/mean_peak_torque": np.mean(self._episode_peak_torques[-window:]),
            "energy/mean_jerk": np.mean(self._episode_jerks[-window:]),
            "energy/mean_max_contact_force": np.mean(self._episode_contact_forces[-window:]),
            "reward/mean_task_reward": np.mean(self._episode_rewards_task[-window:]),
            "reward/mean_energy_penalty": np.mean(self._episode_rewards_energy[-window:]),
            "performance/success_rate": np.mean(self._episode_successes[-window:]),
            "performance/reach_rate": np.mean(self._episode_reached[-window:]),
            "performance/grasp_rate": np.mean(self._episode_grasped[-window:]),
            "performance/mean_episode_length": np.mean(self._episode_lengths[-window:]),
            "performance/total_episodes": n,
        }

        # Also forward SB3's native training metrics (train/*, time/*)
        if hasattr(self, "model") and hasattr(self.model, "logger"):
            sb3_logger = self.model.logger
            if hasattr(sb3_logger, "name_to_value"):
                for key, value in sb3_logger.name_to_value.items():
                    if key.startswith(("train/", "time/", "eco/")):
                        metrics[key] = value

        if wandb.run is not None:
            wandb.log(metrics, step=self.num_timesteps)

        if self.verbose > 0:
            print(
                f"[Step {self.num_timesteps}] "
                f"Success: {metrics['performance/success_rate']:.2%} | "
                f"Grasp: {metrics['performance/grasp_rate']:.2%} | "
                f"Reach: {metrics['performance/reach_rate']:.2%} | "
                f"Jerk: {metrics['energy/mean_jerk']:.2f}"
            )

    def _on_training_end(self):
        """Log final summary."""
        self._log_metrics()


def init_wandb(
    project: str = "energy-aware-manipulation",
    config: Optional[Dict[str, Any]] = None,
    run_name: Optional[str] = None,
    mode: Optional[str] = None,
) -> None:
    """Initialize WandB run.

    Args:
        project: WandB project name.
        config: Configuration dict to log.
        run_name: Optional run name (auto-generated if None).
        mode: 'online', 'offline', or 'disabled'. If None, uses the
              WANDB_MODE environment variable (default: 'online').
    """
    # Respect WANDB_MODE env var when mode is not explicitly set
    if mode is None:
        mode = os.environ.get("WANDB_MODE", "online")

    print(f"[INFO] Running in WANDB {mode} mode")
    wandb.init(
        project=project,
        config=config,
        name=run_name,
        mode=mode,
    )
