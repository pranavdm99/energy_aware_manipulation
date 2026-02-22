"""
evaluate.py — Evaluate a trained agent and collect energy metrics.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/sac_final.zip --task Lift
    python scripts/evaluate.py --checkpoint checkpoints/sac_final.zip --task Lift --save-video
    python scripts/evaluate.py --checkpoint checkpoints/sac_final.zip --task Lift --save-video --n-episodes 5
"""

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set GL backend before importing robosuite/mujoco.
# --save-video without --render needs osmesa for headless offscreen rendering.
# --render needs the default (glfw) for live display windows.
if "--save-video" in sys.argv and "--render" not in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "osmesa")

from stable_baselines3 import SAC
from envs.env_factory import make_env
from agents.sac_agent import evaluate_agent
from utils.metrics import compute_jerk
from utils.constants import MULTITASK_PADDED_DIM, LANGUAGE_EMBEDDING_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.zip)")
    parser.add_argument("--task", type=str, default="Lift")
    parser.add_argument("--energy-weight", type=float, default=0.0,
                        help="Energy weight for evaluation env")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true",
                        help="Enable on-screen rendering (requires display)")
    parser.add_argument("--save-video", action="store_true",
                        help="Record episodes as MP4 videos (offscreen)")
    parser.add_argument("--video-dir", type=str, default="results/videos",
                        help="Directory to save MP4 videos")
    parser.add_argument("--camera", type=str, default="agentview",
                        help="Camera name for video recording (agentview, frontview, etc.)")
    parser.add_argument("--video-size", type=int, nargs=2, default=[480, 480],
                        help="Video frame size (height width)")
    parser.add_argument("--fps", type=int, default=20,
                        help="Video playback FPS")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--language-conditioned", action="store_true")
    parser.add_argument("--descriptor", type=str, default="normally")
    return parser.parse_args()


def render_frame(env, camera_name="agentview"):
    """Render a single frame from the offscreen renderer.

    Walks the wrapper chain to find the robosuite env and calls
    sim.render() to get an RGB numpy array.
    """
    # Walk wrapper chain to the robosuite env
    unwrapped = env
    while hasattr(unwrapped, 'env'):
        unwrapped = unwrapped.env
    # unwrapped is now the robosuite env (or GymWrapper around it)
    if hasattr(unwrapped, 'env'):
        robosuite_env = unwrapped.env
    else:
        robosuite_env = unwrapped

    frame = robosuite_env.sim.render(
        camera_name=camera_name,
        height=480,
        width=480,
    )
    # MuJoCo renders upside-down
    return np.flipud(frame)


def save_video(frames, filepath, fps=20):
    """Save a list of RGB frames as an MP4 video."""
    import imageio
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    imageio.mimwrite(filepath, frames, fps=fps, quality=8)
    print(f"  Video saved: {filepath} ({len(frames)} frames, {len(frames)/fps:.1f}s)")


def detailed_evaluation(model, env, n_episodes=100, verbose=True,
                        record_video=False, camera_name="agentview",
                        video_dir="results/videos", fps=20):
    """Run evaluation with detailed per-episode metrics and optional video recording."""
    all_results = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        frames = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            # Capture frame for video
            if record_video:
                frame = render_frame(env, camera_name)
                frames.append(frame)

        ep_summary = info.get("energy", {}).get("episode_summary", {})

        result = {
            "episode": ep,
            "success": bool(info.get("is_success", False)),
            "total_reward": ep_reward,
            "total_energy": ep_summary.get("total_energy", 0.0),
            "peak_torque": ep_summary.get("peak_torque", 0.0),
            "mean_power": ep_summary.get("mean_power", 0.0),
            "episode_length": ep_summary.get("episode_length", 0),
            "torque_std_per_joint": ep_summary.get("torque_std_per_joint", []),
        }
        all_results.append(result)

        # Save video for this episode
        if record_video and frames:
            success_str = "success" if result["success"] else "fail"
            video_path = os.path.join(
                video_dir, f"ep{ep:03d}_{success_str}.mp4"
            )
            save_video(frames, video_path, fps=fps)

        if verbose and (ep + 1) % 10 == 0:
            successes = [r["success"] for r in all_results]
            energies = [r["total_energy"] for r in all_results]
            print(
                f"Episode {ep + 1}/{n_episodes} | "
                f"SR: {np.mean(successes):.2%} | "
                f"Energy: {np.mean(energies):.2f} ± {np.std(energies):.2f}"
            )

    return all_results


def print_summary(results):
    """Print formatted evaluation summary."""
    successes = [r["success"] for r in results]
    energies = [r["total_energy"] for r in results]
    peak_torques = [r["peak_torque"] for r in results]
    lengths = [r["episode_length"] for r in results]
    rewards = [r["total_reward"] for r in results]

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY ({len(results)} episodes)")
    print(f"{'='*60}")
    print(f"  Success rate:      {np.mean(successes):.2%}")
    print(f"  Total reward:      {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Total energy:      {np.mean(energies):.2f} ± {np.std(energies):.2f}")
    print(f"  Peak torque:       {np.mean(peak_torques):.2f} ± {np.std(peak_torques):.2f}")
    print(f"  Episode length:    {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"{'='*60}\n")


def main():
    args = parse_args()

    # Load model header to check and auto-detect language conditioning
    print(f"Loading model from {args.checkpoint}")
    from stable_baselines3.common.save_util import load_from_zip_file
    data, _, _ = load_from_zip_file(args.checkpoint)
    
    # Auto-detect language conditioning or padding if not explicitly set
    obs_shape = data.get("observation_space").shape
    is_multi_task_padded = False
    
    if obs_shape == (MULTITASK_PADDED_DIM,):
        print(f"  Multi-task padding detected ({obs_shape}).")
        is_multi_task_padded = True
    
    if not args.language_conditioned:
        expected_padded = (MULTITASK_PADDED_DIM + LANGUAGE_EMBEDDING_DIM,)
        if obs_shape == expected_padded:
            print(f"  Multi-task/Padded language model detected ({obs_shape}). Enabling automatically.")
            args.language_conditioned = True
            is_multi_task_padded = True
        elif len(obs_shape) == 1 and obs_shape[0] > LANGUAGE_EMBEDDING_DIM:
            print(f"  Language conditioning suspected in checkpoint ({obs_shape}). Enabling.")
            args.language_conditioned = True

    # Create environment
    env = make_env(
        task=args.task,
        energy_weight=args.energy_weight,
        language_conditioned=args.language_conditioned,
        descriptor=args.descriptor,
        render=args.render,
        camera_name=args.camera if args.save_video else None,
        camera_size=tuple(args.video_size),
        padded_obs_dim=MULTITASK_PADDED_DIM if is_multi_task_padded else None,
    )

    model = SAC.load(args.checkpoint, env=env)

    # Evaluate
    results = detailed_evaluation(
        model, env, args.n_episodes,
        record_video=args.save_video,
        camera_name=args.camera,
        video_dir=args.video_dir,
        fps=args.fps,
    )
    print_summary(results)

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")

    env.close()


if __name__ == "__main__":
    main()
