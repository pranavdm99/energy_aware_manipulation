# Energy-Aware, Language-Conditioned Manipulation

This project involves training a **Franka Panda** arm in **Robosuite** (MuJoCo) to perform manipulation tasks while jointly optimizing **task success** and **mechanical energy efficiency**, with optional **language conditioning** for semantic modulation of motion style.

## Quick Start (Docker)

```bash
# Build the container (first time — may take 10-15 minutes)
docker compose up -d --build

# Enter the container
docker compose exec energy-manip bash

# Verify GPU access
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Verify robosuite
python -c "import robosuite; print(f'robosuite v{robosuite.__version__}')"

# Run unit tests (metrics — no robosuite needed)
python -m pytest tests/test_metrics.py -v

# Run integration tests (requires robosuite)
python -m pytest tests/test_energy_wrapper.py -v

# Smoke test training (5k steps)
python scripts/train.py --task Lift --energy-weight 0.0 --total-timesteps 5000
```

## Project Structure

```
energy_aware_manipulation/
├── configs/
│   ├── default.yaml                  # Base hyperparameters
│   └── ablation_sweep.yaml           # Sweep configurations
├── envs/
│   ├── energy_wrapper.py             # Energy-aware reward wrapper
│   ├── language_wrapper.py           # Language-conditioned wrapper
│   └── env_factory.py                # Environment creation factory
├── agents/
│   └── sac_agent.py                  # SAC training/eval (SB3)
├── scripts/
│   ├── train.py                      # Training entry point
│   ├── evaluate.py                   # Evaluation + metric collection
│   ├── ablation_sweep.py             # Automated sweep runner
│   └── visualize.py                  # Publication-quality plots
├── utils/
│   ├── metrics.py                    # Physics computations
│   ├── language_encoder.py           # Sentence-BERT encoder
│   └── logging_utils.py              # WandB callback
├── Dockerfile
└── docker-compose.yml
```

## Training

WandB defaults to **online** mode. Use `--wandb-mode offline` or set `logging.wandb_mode: "offline"` in config to disable syncing. Ensure you're logged in: `wandb login`.

```bash
# Baseline (no energy penalty)
python scripts/train.py --task Lift --energy-weight 0.0

# Energy-aware (α=0.1)
python scripts/train.py --task Lift --energy-weight 0.1

# Language-conditioned
python scripts/train.py --task Lift --energy-weight 0.05 \
    --language-conditioned --descriptor "gently"

# Full ablation sweep
python scripts/ablation_sweep.py
```

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/Lift_alpha0.0_seed42/sac_final.zip \
    --task Lift --n-episodes 100 --output results/lift_baseline.json

# Generate plots
python scripts/visualize.py --results-dir results/
```

## Architecture

### Energy-Aware Reward
```
r_total = r_task + α · r_energy
r_energy = -mean(|τ_i · ω_i|)   # negative instantaneous power
```

### Language Conditioning
- Sentence-BERT (`all-MiniLM-L6-v2`) encodes descriptors → 384-dim embedding
- Embedding appended to observation vector
- Descriptor maps to energy weight: `"gently" → α=0.1`, `"quickly" → α=0.01`

## Key Metrics
| Metric         | Description                      |
| -------------- | -------------------------------- |
| Success Rate   | Task completion over N episodes  |
| Total Energy   | Σ\|τ·ω\|·Δt over episode         |
| Peak Torque    | max\|τ\| over episode            |
| Jerk           | Σ\|d³x/dt³\| — motion smoothness |
| Episode Length | Steps to completion              |
