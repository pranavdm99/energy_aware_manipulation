
import gymnasium as gym
import numpy as np
from sentence_transformers import SentenceTransformer

class LanguageConditionedWrapper(gym.Wrapper):
    """
    A wrapper that conditions the policy on language descriptors.
    
    1. Encodes the task descriptor (e.g., "gently", "quickly") using Sentence-BERT.
    2. Appends the 384-dimensional embedding to the observation space.
    3. Maps descriptors to energy weights in the info dict (if energy wrapper is present).
    """
    
    # Mapping from descriptors to energy weights (alpha)
    DESCRIPTOR_MAP = {
        "gently": 0.1,    # High penalty -> smooth, slow
        "carefully": 0.1,
        "normally": 0.05, # Balanced
        "quickly": 0.0,   # No penalty -> fast, jerky
        "efficiently": 0.01,
    }

    def __init__(self, env, descriptor="normally", model_name="all-MiniLM-L6-v2", randomize_descriptor=False):
        super().__init__(env)
        self.descriptor = descriptor
        self.randomize_descriptor = randomize_descriptor
        self.model = SentenceTransformer(model_name)
        
        # Cache embeddings to avoid re-encoding every step
        self._embedding_cache = {}
        self._current_embedding = self._get_embedding(descriptor)
        
        # Extend observation space
        # Assuming the base env has a Box observation space
        assert isinstance(env.observation_space, gym.spaces.Box), "Env must have Box observation space"
        
        low = env.observation_space.low
        high = env.observation_space.high
        
        # Embeddings are typically normalized or within [-1, 1], but let's be safe with [-inf, inf] 
        # or just huge bounds for the embedding part.
        embed_dim = self._current_embedding.shape[0]
        
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low, np.full(embed_dim, -np.inf)]),
            high=np.concatenate([high, np.full(embed_dim, np.inf)]),
            dtype=np.float32
        )
        
    def _get_embedding(self, text):
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.model.encode(text)
        return self._embedding_cache[text]

    def set_descriptor(self, descriptor):
        """Update the current task descriptor."""
        if descriptor not in self.DESCRIPTOR_MAP:
             # Default to normally if unknown, or maybe warn?
             pass 
        self.descriptor = descriptor
        self._current_embedding = self._get_embedding(descriptor)

    def reset(self, **kwargs):
        if self.randomize_descriptor:
            # Sample a random descriptor from the map keys
            import random
            descriptors = list(self.DESCRIPTOR_MAP.keys())
            self.descriptor = random.choice(descriptors)
            self._current_embedding = self._get_embedding(self.descriptor)

        obs, info = self.env.reset(**kwargs)
        
        # Retrieve the alpha corresponding to the current descriptor
        alpha = self.DESCRIPTOR_MAP.get(self.descriptor, 0.05)
        
        # Pass this alpha to the EnergyAwareWrapper via info or a direct method if possible.
        # Since we can't easily modify the internal state of a wrapped env directly without unwrapping,
        # we'll put it in info and hope the training loop or a custom callback can use it, 
        # OR we modify EnergyAwareWrapper to look for it.
        # Better approach: EnergyAwareWrapper should be *inside* this wrapper? 
        # Or this wrapper modifies the reward? 
        # Actually, EnergyAwareWrapper computes reward based on self.energy_weight.
        # We can try to set it recursively.
        self._set_energy_weight_recursive(self.env, alpha)
        
        # Make sure info has the descriptor
        info["language"] = {
            "descriptor": self.descriptor,
            "energy_weight": alpha
        }
        
        return self._append_embedding(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Ensure info has language context
        if "language" not in info:
             alpha = self.DESCRIPTOR_MAP.get(self.descriptor, 0.05)
             info["language"] = {
                "descriptor": self.descriptor,
                "energy_weight": alpha
            }
            
        return self._append_embedding(obs), reward, terminated, truncated, info

    def _append_embedding(self, obs):
        return np.concatenate([obs, self._current_embedding]).astype(np.float32)

    def _set_energy_weight_recursive(self, env, weight):
        """Recursively search for EnergyAwareWrapper and update its weight."""
        if hasattr(env, "energy_weight"):
            setattr(env, "energy_weight", weight)
        
        if hasattr(env, "env"):
            self._set_energy_weight_recursive(env.env, weight)
