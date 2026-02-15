"""
LanguageConditionedWrapper — Conditions the policy on natural-language
task descriptors by appending Sentence-BERT embeddings to observations.

Also dynamically adjusts the energy weight of an upstream
EnergyAwareWrapper based on the semantic meaning of the descriptor
(e.g., "gently" -> higher energy penalty, "quickly" -> lower penalty).
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Optional
from utils.language_encoder import LanguageEncoder


# Default mapping from descriptors to energy weight modifiers
DEFAULT_DESCRIPTOR_MAP: Dict[str, float] = {
    "gently": 0.1,
    "carefully": 0.1,
    "softly": 0.1,
    "normally": 0.05,
    "steadily": 0.05,
    "quickly": 0.01,
    "efficiently": 0.01,
    "fast": 0.01,
}


class LanguageConditionedWrapper(gym.Wrapper):
    """Augments observations with language embeddings and modulates energy weight.

    Args:
        env: Upstream environment (should be an EnergyAwareWrapper).
        descriptor: Initial task descriptor string.
        descriptor_map: Dict mapping descriptor keywords to energy weights.
        model_name: Sentence-BERT model to use for encoding.
        randomize_descriptor: If True, randomly sample descriptors at reset.
    """

    def __init__(
        self,
        env: gym.Env,
        descriptor: str = "normally",
        descriptor_map: Optional[Dict[str, float]] = None,
        model_name: str = "all-MiniLM-L6-v2",
        randomize_descriptor: bool = False,
    ):
        super().__init__(env)
        self.encoder = LanguageEncoder(model_name=model_name)
        self.descriptor_map = descriptor_map or DEFAULT_DESCRIPTOR_MAP
        self.randomize_descriptor = randomize_descriptor

        # Encode initial descriptor
        self._descriptor = descriptor
        self._embedding = self.encoder.encode(descriptor)
        self._embedding_dim = len(self._embedding)

        # Expand observation space to include language embedding
        low = self.observation_space.low
        high = self.observation_space.high
        new_low = np.concatenate([
            low, -np.ones(self._embedding_dim, dtype=np.float32)
        ])
        new_high = np.concatenate([
            high, np.ones(self._embedding_dim, dtype=np.float32)
        ])
        self.observation_space = gym.spaces.Box(
            low=new_low, high=new_high, dtype=np.float32
        )

        # Apply energy weight from descriptor
        self._apply_descriptor_energy_weight(descriptor)

    def _apply_descriptor_energy_weight(self, descriptor: str):
        """Set the energy weight of the upstream EnergyAwareWrapper based on descriptor."""
        # Find matching keyword in descriptor
        energy_weight = self.descriptor_map.get("normally", 0.05)  # default
        descriptor_lower = descriptor.lower()
        for keyword, weight in self.descriptor_map.items():
            if keyword in descriptor_lower:
                energy_weight = weight
                break

        # Walk wrapper chain to find EnergyAwareWrapper and set its weight
        current = self.env
        while current is not None:
            if hasattr(current, "energy_weight"):
                current.energy_weight = energy_weight
                break
            current = getattr(current, "env", None)

    def set_descriptor(self, descriptor: str):
        """Change the task descriptor (and re-encode + adjust energy weight)."""
        self._descriptor = descriptor
        self._embedding = self.encoder.encode(descriptor)
        self._apply_descriptor_energy_weight(descriptor)

    def reset(self, **kwargs):
        """Reset with optional descriptor randomization."""
        if self.randomize_descriptor:
            descriptors = list(self.descriptor_map.keys())
            chosen = np.random.choice(descriptors)
            self.set_descriptor(chosen)

        obs, info = self.env.reset(**kwargs)

        # Append language embedding to observation
        obs = np.concatenate([obs, self._embedding]).astype(np.float32)

        # Add language info
        info["language"] = {
            "descriptor": self._descriptor,
            "embedding_dim": self._embedding_dim,
        }

        return obs, info

    def step(self, action):
        """Step with language-augmented observations."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Append language embedding
        obs = np.concatenate([obs, self._embedding]).astype(np.float32)

        # Add language info
        info["language"] = {
            "descriptor": self._descriptor,
            "embedding_dim": self._embedding_dim,
        }

        return obs, reward, terminated, truncated, info
