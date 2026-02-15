"""
EnergyAwareWrapper — Gymnasium wrapper that augments robosuite environments
with mechanical energy penalties and torque monitoring.

Reward is modified as:
    r_total = r_task + alpha * r_energy
where:
    r_energy = -mean(|tau_i * omega_i|)  (negative instantaneous power)

This encourages the agent to find energy-efficient motions while still
completing the manipulation task.
"""

import gymnasium as gym
import numpy as np
from typing import Optional


class EnergyAwareWrapper(gym.Wrapper):
    """Wraps a robosuite GymWrapper environment with energy-aware reward shaping.

    Extracts joint torques and velocities from MuJoCo sim data at each step,
    computes mechanical power and cumulative energy, modifies the reward with
    an energy penalty, and logs all energy metrics in the info dict.

    Args:
        env: A gymnasium-compatible robosuite environment (via GymWrapper).
        energy_weight: Scalar alpha for energy penalty (0 = no penalty).
        normalize_by_dof: If True, normalize energy penalty by number of DOFs.
        include_in_obs: If True, append torque features to observations.
    """

    def __init__(
        self,
        env: gym.Env,
        energy_weight: float = 0.0,
        normalize_by_dof: bool = True,
        include_in_obs: bool = False,
    ):
        super().__init__(env)
        self.energy_weight = energy_weight
        self.normalize_by_dof = normalize_by_dof
        self.include_in_obs = include_in_obs

        # Access the underlying robosuite env through the wrapper chain
        self._robosuite_env = self._get_robosuite_env(env)

        # Determine robot DOF count from the robosuite env
        self._n_dof = self._robosuite_env.robots[0].dof
        self._dt = self._robosuite_env.control_timestep

        # If including torque in obs, expand observation space
        if self.include_in_obs:
            low = self.observation_space.low
            high = self.observation_space.high
            # Append: [torques (n_dof), velocities (n_dof), inst_power (1)]
            extra_dim = 2 * self._n_dof + 1
            new_low = np.concatenate([low, -np.inf * np.ones(extra_dim)])
            new_high = np.concatenate([high, np.inf * np.ones(extra_dim)])
            self.observation_space = gym.spaces.Box(
                low=new_low.astype(np.float32),
                high=new_high.astype(np.float32),
                dtype=np.float32,
            )

        # Episode accumulators
        self._episode_energy = 0.0
        self._episode_peak_torque = 0.0
        self._episode_torques = []
        self._episode_positions = []
        self._step_count = 0

    def _get_robosuite_env(self, env):
        """Walk the wrapper chain to find the robosuite base environment."""
        current = env
        while hasattr(current, "env"):
            if hasattr(current, "robots"):
                return current
            current = current.env
        if hasattr(current, "robots"):
            return current
        raise ValueError(
            "Could not find robosuite environment in wrapper chain. "
            "Ensure this wraps a robosuite GymWrapper environment."
        )

    def _get_torques_and_velocities(self):
        """Extract joint torques and velocities from MuJoCo sim data."""
        sim = self._robosuite_env.sim
        robot = self._robosuite_env.robots[0]

        # Joint indices for this robot
        joint_indices = [
            sim.model.joint_name2id(joint) for joint in robot.robot_joints
        ]

        # Actuator torques (what the robot is actually commanding)
        # qfrc_actuator gives the force/torque applied by actuators
        torques = np.array([sim.data.qfrc_actuator[i] for i in joint_indices])

        # Joint velocities
        velocities = np.array([sim.data.qvel[i] for i in joint_indices])

        return torques, velocities

    def _compute_energy_penalty(self, torques, velocities):
        """Compute energy penalty from torques and velocities.

        Returns:
            penalty: Negative mean instantaneous power (scalar).
            power: Instantaneous power per joint (array).
        """
        # Instantaneous power per joint: P_i = |tau_i * omega_i|
        power_per_joint = np.abs(torques * velocities)
        total_power = np.sum(power_per_joint)

        if self.normalize_by_dof:
            penalty = -total_power / self._n_dof
        else:
            penalty = -total_power

        return penalty, power_per_joint

    def reset(self, **kwargs):
        """Reset environment and energy accumulators."""
        obs, info = self.env.reset(**kwargs)

        # Reset episode accumulators
        self._episode_energy = 0.0
        self._episode_peak_torque = 0.0
        self._episode_torques = []
        self._episode_positions = []
        self._step_count = 0

        # Add initial energy info
        info["energy"] = {
            "cumulative_energy": 0.0,
            "peak_torque": 0.0,
            "step_power": 0.0,
            "energy_weight": self.energy_weight,
        }

        if self.include_in_obs:
            extra = np.zeros(2 * self._n_dof + 1, dtype=np.float32)
            obs = np.concatenate([obs, extra])

        return obs, info

    def step(self, action):
        """Step the environment and compute energy metrics."""
        obs, reward_task, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        # --- Extract physics data ---
        torques, velocities = self._get_torques_and_velocities()

        # --- Compute energy penalty ---
        penalty, power_per_joint = self._compute_energy_penalty(
            torques, velocities
        )
        instantaneous_power = np.sum(power_per_joint)

        # --- Update accumulators ---
        step_energy = instantaneous_power * self._dt
        self._episode_energy += step_energy
        self._episode_peak_torque = max(
            self._episode_peak_torque, np.max(np.abs(torques))
        )
        self._episode_torques.append(torques.copy())
        self._episode_positions.append(
            self._robosuite_env.sim.data.qpos[: self._n_dof].copy()
        )

        # --- Modify reward ---
        reward_total = reward_task + self.energy_weight * penalty

        # --- Populate info ---
        info["energy"] = {
            "reward_task": reward_task,
            "reward_energy_penalty": self.energy_weight * penalty,
            "reward_total": reward_total,
            "step_power": instantaneous_power,
            "step_energy": step_energy,
            "cumulative_energy": self._episode_energy,
            "peak_torque": self._episode_peak_torque,
            "torques": torques.tolist(),
            "power_per_joint": power_per_joint.tolist(),
            "energy_weight": self.energy_weight,
        }

        # --- Augment observation if requested ---
        if self.include_in_obs:
            extra = np.concatenate([
                torques,
                velocities,
                [instantaneous_power],
            ]).astype(np.float32)
            obs = np.concatenate([obs, extra])

        # --- On episode end, add summary metrics ---
        if terminated or truncated:
            ep_torques = np.array(self._episode_torques)
            info["energy"]["episode_summary"] = {
                "total_energy": self._episode_energy,
                "peak_torque": self._episode_peak_torque,
                "mean_power": (
                    self._episode_energy / (self._step_count * self._dt)
                    if self._step_count > 0
                    else 0.0
                ),
                "torque_std_per_joint": ep_torques.std(axis=0).tolist(),
                "torque_mean_per_joint": ep_torques.mean(axis=0).tolist(),
                "episode_length": self._step_count,
            }

        return obs, reward_total, terminated, truncated, info
