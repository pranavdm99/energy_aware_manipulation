"""
Environment factory — creates fully wrapped robosuite environments
ready for training with SB3.

Pipeline: robosuite.make() -> GymWrapper -> EnergyAwareWrapper -> [LanguageConditionedWrapper]
"""

import gymnasium as gym
import robosuite as suite
from robosuite.wrappers import GymWrapper

from envs.energy_wrapper import EnergyAwareWrapper
from envs.language_wrapper import LanguageConditionedWrapper


def make_env(
    task: str = "Lift",
    robots: str = "Panda",
    controller: str = "OSC_POSE",
    horizon: int = 500,
    reward_shaping: bool = True,
    energy_weight: float = 0.0,
    normalize_by_dof: bool = True,
    include_energy_in_obs: bool = False,
    language_conditioned: bool = False,
    descriptor: str = "normally",
    descriptor_map: dict = None,
    randomize_descriptor: bool = False,
    language_model: str = "all-MiniLM-L6-v2",
    render: bool = False,
    camera_name: str = None,
    camera_size: tuple = (480, 480),
    seed: int = None,
) -> gym.Env:
    """Create a fully wrapped robosuite environment.

    Args:
        task: Robosuite task name (Lift, PickPlace, Door, NutAssemblySingle, etc.)
        robots: Robot name (default: Panda).
        controller: Controller type (OSC_POSE for operational space control).
        horizon: Max steps per episode.
        reward_shaping: Use dense reward shaping.
        energy_weight: Alpha for energy penalty (0 = no penalty).
        normalize_by_dof: Normalize energy penalty by DOF count.
        include_energy_in_obs: Append torque features to observations.
        language_conditioned: Enable language conditioning.
        descriptor: Initial task descriptor.
        descriptor_map: Dict mapping descriptors to energy weights.
        randomize_descriptor: Randomly sample descriptor at each reset.
        language_model: Sentence-BERT model name.
        render: Enable on-screen rendering.
        camera_name: Camera for offscreen rendering (e.g. 'agentview', 'frontview').
                     If set, enables offscreen rendering for video recording.
        camera_size: (height, width) tuple for offscreen renders.
        seed: Random seed.

    Returns:
        A gymnasium-compatible environment with energy and/or language wrappers.
    """
    # --- Create robosuite environment ---
    # robosuite v1.5+ auto-loads the default controller config for each robot
    use_offscreen = camera_name is not None
    robosuite_env = suite.make(
        env_name=task,
        robots=robots,
        has_renderer=render,
        has_offscreen_renderer=use_offscreen,
        use_camera_obs=False,
        horizon=horizon,
        reward_shaping=reward_shaping,
        **({
            "camera_names": camera_name,
            "camera_heights": camera_size[0],
            "camera_widths": camera_size[1],
        } if use_offscreen else {}),
    )

    # --- Wrap with GymWrapper for gymnasium API ---
    gym_env = GymWrapper(robosuite_env)

    # --- Wrap with EnergyAwareWrapper ---
    energy_env = EnergyAwareWrapper(
        env=gym_env,
        energy_weight=energy_weight,
        normalize_by_dof=normalize_by_dof,
        include_in_obs=include_energy_in_obs,
    )

    # --- Optionally wrap with LanguageConditionedWrapper ---
    if language_conditioned:
        final_env = LanguageConditionedWrapper(
            env=energy_env,
            descriptor=descriptor,
            descriptor_map=descriptor_map,
            model_name=language_model,
            randomize_descriptor=randomize_descriptor,
        )
    else:
        final_env = energy_env

    return final_env


def make_env_from_config(config: dict) -> gym.Env:
    """Create environment from a configuration dictionary (loaded from YAML).

    Args:
        config: Full configuration dict with 'environment', 'energy',
                'language' sections.

    Returns:
        A fully wrapped gymnasium environment.
    """
    env_cfg = config.get("environment", {})
    energy_cfg = config.get("energy", {})
    lang_cfg = config.get("language", {})

    return make_env(
        task=env_cfg.get("task", "Lift"),
        robots=env_cfg.get("robots", "Panda"),
        controller=env_cfg.get("controller", "OSC_POSE"),
        horizon=env_cfg.get("horizon", 500),
        reward_shaping=env_cfg.get("reward_shaping", True),
        energy_weight=energy_cfg.get("weight", 0.0),
        normalize_by_dof=energy_cfg.get("normalize_by_dof", True),
        include_energy_in_obs=energy_cfg.get("include_in_obs", False),
        language_conditioned=lang_cfg.get("enabled", False),
        descriptor=lang_cfg.get("descriptor", "normally"),
        descriptor_map=lang_cfg.get("descriptor_map", None),
        randomize_descriptor=lang_cfg.get("randomize_descriptor", False),
        language_model=lang_cfg.get("model", "all-MiniLM-L6-v2"),
    )
