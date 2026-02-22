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

    # ---------------------------------------------------------------
    # Door-specific contact detection helpers
    # ---------------------------------------------------------------
    # Geom names discovered via scripts/inspect_door_env.py
    _DOOR_HANDLE_GEOMS = frozenset({
        "Door_handle",
        "Door_handle_base",
        "Door_latch",
        "Door_latch_tip",
    })
    _GRIPPER_PAD_GEOMS = frozenset({
        "gripper0_right_finger1_pad_collision",
        "gripper0_right_finger2_pad_collision",
    })

    def _check_door_handle_contact(self, sim) -> bool:
        """Return True if either finger pad is in contact with the door handle.

        Iterates the MuJoCo contact array and looks for any pair where one
        geom is a gripper pad and the other is a door handle geom.
        """
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            g1 = sim.model.geom_id2name(c.geom1)
            g2 = sim.model.geom_id2name(c.geom2)
            if (g1 in self._GRIPPER_PAD_GEOMS and g2 in self._DOOR_HANDLE_GEOMS) or \
               (g2 in self._GRIPPER_PAD_GEOMS and g1 in self._DOOR_HANDLE_GEOMS):
                return True
        return False

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
        
        # Get gripper position
        try:
            # Try to find the gripper site
            # Common names: gripper0_grip_site, gripper0_right_grip_site, grip_site
            grip_site_name = "gripper0_right_grip_site"
            if grip_site_name not in sim.model.site_names:
                if "gripper0_grip_site" in sim.model.site_names: # For older robosuite
                    grip_site_name = "gripper0_grip_site"
                else:
                    for name in sim.model.site_names:
                        if "grip_site" in name:
                            grip_site_name = name
                            break
            
            gripper_pos = sim.data.site_xpos[sim.model.site_name2id(grip_site_name)]

            # Identify target object position based on task type
            # We check the unwrapped env for specific attributes
            base_env = self.unwrapped
            target_pos = None

            # 1. Lift (cube)
            if hasattr(base_env, "cube"):
                 # Lift task
                 # cube.root_body gives the body name usually? No, it gives ID or name depending on version.
                 # verification script showed obj_body_id={'cube': 26}.
                 # We can use sim.data.body_xpos with body name "cube_main"
                 obj_name = "cube_main" if "cube_main" in sim.model.body_names else "cube"
                 target_pos = sim.data.body_xpos[sim.model.body_name2id(obj_name)]
            
            # 2. Door — use handle site for reach, MuJoCo contacts for grasp
            elif hasattr(base_env, "door"):
                # Get handle position from the reliable site id
                if hasattr(base_env, "door_handle_site_id"):
                    target_pos = sim.data.site_xpos[base_env.door_handle_site_id]
                elif "Door_handle" in sim.model.site_names:
                    target_pos = sim.data.site_xpos[
                        sim.model.site_name2id("Door_handle")
                    ]

            # 3. NutAssembly
            elif hasattr(base_env, "nuts"):
                for nut_name in ["SquareNut_main", "RoundNut_main"]:
                    if nut_name in sim.model.body_names:
                        target_pos = sim.data.body_xpos[
                            sim.model.body_name2id(nut_name)
                        ]
                        break

            # 4. PickPlace (objects)
            elif hasattr(base_env, "objects"):
                min_dist = float("inf")
                best_pos = None
                for obj in base_env.objects:
                    name = obj.name + "_main"
                    if name in sim.model.body_names:
                        pos = sim.data.body_xpos[sim.model.body_name2id(name)]
                        dist = np.linalg.norm(pos - gripper_pos)
                        if dist < min_dist:
                            min_dist = dist
                            best_pos = pos
                target_pos = best_pos

            # Fallback
            if target_pos is None:
                obj_name = "cube_main" if "cube_main" in sim.model.body_names else "cube"
                if obj_name in sim.model.body_names:
                    target_pos = sim.data.body_xpos[sim.model.body_name2id(obj_name)]

            if target_pos is not None:
                dist = np.linalg.norm(target_pos - gripper_pos)

                if hasattr(base_env, "door"):
                    # Door: reach = gripper close to handle site
                    #        grasp = finger pad physically contacting handle geom
                    is_reached = dist < 0.06
                    is_grasped = self._check_door_handle_contact(sim)
                else:
                    # Lift / PickPlace: reach by distance, grasp by height
                    is_reached = dist < 0.05
                    is_grasped = is_reached and (target_pos[2] > 0.82)
            else:
                is_reached = False
                is_grasped = False
                
        except (ValueError, KeyError, AttributeError):
            is_reached = False
            is_grasped = False

        # --- Door shaped bonus: reward reaching and contacting the handle ---
        # This prevents body-pushing reward hacking by incentivising gripper use.
        # Bonuses are small (0.3 / 0.5) relative to the native door-angle reward
        # (~2.5 max) so they shape without distorting the reward scale.
        door_bonus = 0.0
        if hasattr(self._robosuite_env, "door"):
            if is_reached:
                door_bonus += 0.3
            if is_grasped:
                door_bonus += 0.5

        # --- Modify reward ---
        reward_total = reward_task + door_bonus + self.energy_weight * penalty

        # --- Populate info ---
        info["energy"] = {
            "reward_task": reward_task,
            "reward_door_bonus": door_bonus,
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
