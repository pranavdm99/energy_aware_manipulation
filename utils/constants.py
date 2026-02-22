
"""
Constants — Shared configuration and magic numbers for the project.
"""

# --- Observation Space Dimensions ---
BASE_OBS_DIM = {
    "Lift": 60,
    "PickPlace": 106,
    "Door": 64,
    "NutAssemblySingle": 65
}

# The target dimension for MultiTask padding
# We pad base observations to this value before adding language embeddings
MULTITASK_PADDED_DIM = 110

# Language model dimensions
LANGUAGE_EMBEDDING_DIM = 384

# --- Energy Aware Manipulation ---
# Mapping of descriptors to Lagrangian energy budgets (epsilon)
# NOTE: budgets are task-specific — different tasks have different energy scales.

# Lift task: agent moves ~400–1400 units of energy. Budget reflects this range.
ENERGY_BUDGET_MAP = {
    "gently":      150.0,
    "normally":    300.0,
    "carefully":   150.0,
    "efficiently": 150.0,
    "quickly":     600.0,
}

# Door task: smaller workspace + pulling motion uses less energy than Lift.
# Budgets tuned from ~100–500 range observed in Door training.
ENERGY_BUDGET_MAP_DOOR = {
    "gently":      100.0,
    "normally":    250.0,
    "carefully":   100.0,
    "efficiently": 120.0,
    "quickly":     500.0,
}

# Lookup by task name for use in env_factory / train.py
ENERGY_BUDGET_BY_TASK = {
    "Lift":      ENERGY_BUDGET_MAP,
    "Door":      ENERGY_BUDGET_MAP_DOOR,
    "PickPlace": ENERGY_BUDGET_MAP,       # placeholder — tune after Door run
}


# Default energy weights (alpha) for fixed-penalty baseline
DEFAULT_ENERGY_MAP = {
    "gently": 0.1,
    "normally": 0.05,
    "carefully": 0.1,
    "efficiently": 0.1,
    "quickly": 0.01,
}

# --- Training / Optimization ---
LAGRANGIAN_CLIP_MAX = 50.0
DEFAULT_NET_ARCH = [256, 256]
