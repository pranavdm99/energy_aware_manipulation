"""
Physics-based metrics for evaluating manipulation energy efficiency.

All computations are vectorized with NumPy for performance.
"""

import numpy as np
from typing import Dict, List


def compute_power(
    torques: np.ndarray, velocities: np.ndarray
) -> np.ndarray:
    """Compute instantaneous mechanical power per joint.

    P_i(t) = |tau_i(t) * omega_i(t)|

    Args:
        torques: Joint torques, shape (n_joints,) or (T, n_joints).
        velocities: Joint angular velocities, same shape as torques.

    Returns:
        Absolute power per joint, same shape as inputs.
    """
    return np.abs(torques * velocities)


def compute_energy(
    torques: np.ndarray, velocities: np.ndarray, dt: float
) -> float:
    """Compute total mechanical energy over a trajectory.

    E = sum_t sum_i |tau_i(t) * omega_i(t)| * dt

    Args:
        torques: Shape (T, n_joints).
        velocities: Shape (T, n_joints).
        dt: Simulation timestep.

    Returns:
        Total energy (scalar).
    """
    power = compute_power(torques, velocities)
    return float(np.sum(power) * dt)


def compute_peak_torque(torques: np.ndarray) -> float:
    """Compute peak absolute torque across all joints and timesteps.

    Args:
        torques: Shape (T, n_joints).

    Returns:
        Maximum absolute torque value.
    """
    return float(np.max(np.abs(torques)))


def compute_jerk(positions: np.ndarray, dt: float) -> float:
    """Compute jerk-based smoothness metric.

    Jerk is the third derivative of position. Lower jerk = smoother motion.
    Uses finite differences: jerk(t) = (a(t+1) - a(t)) / dt
    where a(t) = (v(t+1) - v(t)) / dt

    Args:
        positions: Joint positions, shape (T, n_joints).
        dt: Timestep.

    Returns:
        Mean absolute jerk (scalar, lower = smoother).
    """
    if len(positions) < 4:
        return 0.0

    # First derivative (velocity)
    vel = np.diff(positions, axis=0) / dt
    # Second derivative (acceleration)
    acc = np.diff(vel, axis=0) / dt
    # Third derivative (jerk)
    jerk = np.diff(acc, axis=0) / dt

    return float(np.mean(np.abs(jerk)))


def compute_torque_distribution(torques: np.ndarray) -> Dict[str, List[float]]:
    """Compute per-joint torque statistics over an episode.

    Args:
        torques: Shape (T, n_joints).

    Returns:
        Dict with per-joint mean, std, min, max, and RMS torques.
    """
    return {
        "mean": np.mean(torques, axis=0).tolist(),
        "std": np.std(torques, axis=0).tolist(),
        "min": np.min(torques, axis=0).tolist(),
        "max": np.max(torques, axis=0).tolist(),
        "rms": np.sqrt(np.mean(torques**2, axis=0)).tolist(),
    }


def compute_all_metrics(
    torques: np.ndarray,
    velocities: np.ndarray,
    positions: np.ndarray,
    dt: float,
) -> Dict[str, float]:
    """Compute all evaluation metrics for a trajectory.

    Args:
        torques: Shape (T, n_joints).
        velocities: Shape (T, n_joints).
        positions: Shape (T, n_joints).
        dt: Timestep.

    Returns:
        Dict with total_energy, peak_torque, mean_power, jerk, episode_length.
    """
    total_energy = compute_energy(torques, velocities, dt)
    peak_torque = compute_peak_torque(torques)
    mean_power = total_energy / (len(torques) * dt) if len(torques) > 0 else 0.0
    jerk = compute_jerk(positions, dt)

    return {
        "total_energy": total_energy,
        "peak_torque": peak_torque,
        "mean_power": mean_power,
        "jerk": jerk,
        "episode_length": len(torques),
    }
