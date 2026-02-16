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
        self._episode_jerk = 0.0
        self._episode_max_contact_force = 0.0
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
        sim = self._robosuite_env.sim

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
        
        # Current joint positions
        current_pos = sim.data.qpos[: self._n_dof].copy()
        self._episode_positions.append(current_pos)

        # --- Compute Jerk ---
        jerk = 0.0
        if len(self._episode_positions) >= 4:
            p = self._episode_positions
            j_vec = (p[-1] - 3*p[-2] + 3*p[-3] - p[-4]) / (self._dt ** 3)
            jerk = np.mean(np.abs(j_vec))
        self._episode_jerk += jerk

        # --- Extract Contact Forces ---
        max_contact_force = 0.0
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            # Use efc_force which is directly available in MjData
            # Note: contact.efc_address points to the start of the force vector in efc_force
            if contact.efc_address >= 0:
                force = sim.data.efc_force[contact.efc_address : contact.efc_address + 3]
                max_contact_force = max(max_contact_force, np.linalg.norm(force))
        self._episode_max_contact_force = max(self._episode_max_contact_force, max_contact_force)

        # --- Track Stage Progress (for Lift task) ---
        is_success = bool(self._robosuite_env._check_success())
        
        # Get cube and gripper positions
        try:
            # Try to find the object body - Lift uses 'cube_main'
            obj_name = "cube_main" if "cube_main" in sim.model.body_names else "cube"
            if obj_name not in sim.model.body_names:
                for name in sim.model.body_names:
                    if "cube" in name or "object" in name:
                        obj_name = name
                        break
            
            cube_pos = sim.data.body_xpos[sim.model.body_name2id(obj_name)]
            gripper_pos = sim.data.site_xpos[sim.model.site_name2id("gripper0_grip_site")]
            dist = np.linalg.norm(cube_pos - gripper_pos)
            
            is_reached = dist < 0.05
            is_grasped = is_reached and (cube_pos[2] > 0.82)
        except (ValueError, KeyError):
            is_reached = False
            is_grasped = False

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
            "step_jerk": jerk,
            "max_contact_force": max_contact_force,
            "is_reached": bool(is_reached),
            "is_grasped": bool(is_grasped),
            "is_success": is_success,
            "energy_weight": self.energy_weight,
        }

        # Ensure top-level info also has metrics for callbacks
        info["is_success"] = is_success
        info["is_reached"] = is_reached
        info["is_grasped"] = is_grasped

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
                "mean_jerk": self._episode_jerk / self._step_count if self._step_count > 0 else 0.0,
                "max_contact_force": self._episode_max_contact_force,
                "episode_length": self._step_count,
            }

        return obs, reward_total, terminated, truncated, info
