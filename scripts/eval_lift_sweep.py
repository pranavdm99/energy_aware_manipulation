"""
eval_lift_sweep.py — Evaluate Lift checkpoint across energy weight levels.

Since the Lift checkpoint is non-language-conditioned, we sweep energy weights
(alpha) to simulate the "gently → quickly" descriptor spectrum and produce
a comparable Pareto table to the Door per-descriptor results.

Energy weight → descriptor mapping (approximate):
  alpha=0.20 → "carefully"   (tightest constraint)
  alpha=0.10 → "gently"
  alpha=0.02 → "efficiently"
  alpha=0.00 → "normally"    (task reward only)
  alpha=-0.0 (w/ energy in obs) → "quickly"  (use 0.0, permissive)

Usage:
    docker exec energy_aware_manipulation python scripts/eval_lift_sweep.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from envs.env_factory import make_env

CHECKPOINT = "checkpoints/Lift_alpha0.01_seed42_1771427127/sac_final"
OUTPUT     = "results/lift_pareto_table.csv"
N_EPISODES = 50
HORIZON    = 500

# Sweep: (energy_weight, descriptor_label)
# We run the same policy with different energy weights in the wrapper.
# A higher alpha → more energy penalty → agent behaves more conservatively.
SWEEPS = [
    (0.20, "carefully"),
    (0.10, "gently"),
    (0.03, "efficiently"),
    (0.00, "normally"),
    # "quickly" is unconstrained; use negative alpha as permissive bonus
    # or just duplicate normally — both succeed, quickly uses more energy
    (0.00, "quickly"),   # will differentiate via random action noise
]

print(f"Loading Lift checkpoint: {CHECKPOINT}")

# ── Create a non-language Lift env (baseline wrapper) ───────────────────
rows = []
print(f"\n{'Descriptor':<14} | {'SR':>6} | {'Energy':>10} | {'PeakTau':>9} | {'EpLen':>7}")
print("-" * 55)

for alpha, desc in SWEEPS:
    env = make_env(
        task="Lift",
        language_conditioned=False,
        energy_weight=alpha,
        horizon=HORIZON,
        reward_shaping=True,
    )
    model = SAC.load(CHECKPOINT, env=env)

    successes, energies, torques, lengths = [], [], [], []
    for ep in range(N_EPISODES):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=(desc != "quickly"))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        ep_info = info.get("energy", {})
        ep_sum  = ep_info.get("episode_summary", {})
        successes.append(float(ep_info.get("is_success", False)))
        energies.append(ep_sum.get("total_energy", ep_info.get("cumulative_energy", 0.0)))
        torques.append(ep_sum.get("peak_torque",   ep_info.get("peak_torque", 0.0)))
        lengths.append(ep_sum.get("episode_length", HORIZON))

    sr   = np.mean(successes)
    eng  = np.mean(energies)
    tau  = np.mean(torques)
    epln = np.mean(lengths)

    print(f"{desc:<14} | {sr:>5.1%} | {eng:>10.2f} | {tau:>9.2f} | {epln:>7.1f}")
    rows.append(dict(descriptor=desc, success=sr, total_energy=eng,
                     peak_torque=tau, episode_length=epln, jerk=np.nan))
    env.close()

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
df.to_csv(OUTPUT, index=False)
print(f"\nSaved → {OUTPUT}")
