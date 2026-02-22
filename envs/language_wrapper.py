
import gymnasium as gym
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.constants import ENERGY_BUDGET_MAP, ENERGY_BUDGET_BY_TASK, DEFAULT_ENERGY_MAP

class LanguageConditionedWrapper(gym.Wrapper):
    """
    A wrapper that conditions the policy on language descriptors.
    
    1. Encodes the task descriptor (e.g., "gently", "quickly") using Sentence-BERT.
    2. Appends the 384-dimensional embedding to the observation space.
    3. Maps descriptors to energy weights in the info dict (if energy wrapper is present).
    """
    
    # Mapping from descriptors to energy weights (alpha)
    DESCRIPTOR_MAP = DEFAULT_ENERGY_MAP

    # Default (Lift-scale) budget map; overridden per-task in reset()
    ENERGY_BUDGET_MAP = ENERGY_BUDGET_MAP

    def __init__(self, env, descriptor="normally", model_name="all-MiniLM-L6-v2", randomize_descriptor=False, pre_calculated_embeddings=None):
        super().__init__(env)
        self.descriptor = descriptor
        self.randomize_descriptor = randomize_descriptor
        
        # Cache embeddings to avoid re-encoding every step
        self._embedding_cache = pre_calculated_embeddings or {}
        
        if not self._embedding_cache:
            print(f"[LanguageConditionedWrapper] No pre-calculated embeddings. Loading {model_name}...")
            self.model = SentenceTransformer(model_name)
        else:
            self.model = None
            
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
            if self.model is not None:
                self._embedding_cache[text] = self.model.encode(text)
            else:
                raise ValueError(f"Descriptor '{text}' not in pre-calculated embeddings and no language model loaded!")
        return self._embedding_cache[text]

    def set_descriptor(self, descriptor):
        """Update the current task descriptor."""
        if descriptor not in self.DESCRIPTOR_MAP:
             # Default to normally if unknown, or maybe warn?
             pass 
        self.descriptor = descriptor
        self._current_embedding = self._get_embedding(descriptor)

    def _get_task_budget_map(self):
        """Return the ECO budget map appropriate for the current task."""
        # Walk wrapper chain to find the robosuite task name
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        task_name = getattr(env, "name", None) or getattr(env, "env_name", None)
        if task_name and task_name in ENERGY_BUDGET_BY_TASK:
            return ENERGY_BUDGET_BY_TASK[task_name]
        return self.ENERGY_BUDGET_MAP  # default (Lift-scale)

    def reset(self, **kwargs):
        if self.randomize_descriptor:
            import random
            descriptors = list(self.DESCRIPTOR_MAP.keys())
            self.descriptor = random.choice(descriptors)
            self._current_embedding = self._get_embedding(self.descriptor)

        obs, info = self.env.reset(**kwargs)

        # ECO Integration: Check if we have an adaptive weight from the registry
        try:
            from utils.constrained_rl import LagrangianRegistry
            alpha = LagrangianRegistry.get_weight(self.descriptor, default=None)
        except ImportError:
            alpha = None

        if alpha is None:
            alpha = self.DESCRIPTOR_MAP.get(self.descriptor, 0.05)

        # Use task-appropriate budget map
        task_budget_map = self._get_task_budget_map()
        budget = task_budget_map.get(self.descriptor, 150.0)
        
        # Pass this alpha to the EnergyAwareWrapper
        self._set_energy_weight_recursive(self.env, alpha)
        
        # Make sure info has the descriptor
        info["language"] = {
            "descriptor": self.descriptor,
            "energy_weight": alpha,
            "energy_budget": budget
        }
        
        return self._append_embedding(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Periodically re-sync weight from registry if in ECO mode
        # This ensures the step reward computation uses the latest lambda
        from utils.constrained_rl import LagrangianRegistry
        alpha = LagrangianRegistry.get_weight(self.descriptor, default=None)
        if alpha is not None:
             self._set_energy_weight_recursive(self.env, alpha)
        else:
             alpha = self.DESCRIPTOR_MAP.get(self.descriptor, 0.05)

        # Ensure info has language context
        if "language" not in info:
             budget = self.ENERGY_BUDGET_MAP.get(self.descriptor, 150.0)
             info["language"] = {
                "descriptor": self.descriptor,
                "energy_weight": alpha,
                "energy_budget": budget
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
