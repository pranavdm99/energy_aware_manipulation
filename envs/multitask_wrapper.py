
import gymnasium as gym
import numpy as np

class MultiTaskWrapper(gym.Wrapper):
    """
    A wrapper that unifies observation spaces across multiple robosuite tasks
    by padding them to a fixed maximum dimension.
    
    This allows a single policy to be trained on multiple environments with 
    different underlying observation structures.
    """
    def __init__(self, env, target_dim=110):
        super().__init__(env)
        self.target_dim = target_dim
        
        # Original observation space
        assert isinstance(env.observation_space, gym.spaces.Box), "Env must have Box observation space"
        self.original_obs_space = env.observation_space
        
        if self.original_obs_space.shape[0] > target_dim:
            raise ValueError(f"Task observation dimension {self.original_obs_space.shape[0]} "
                             f"exceeds target dimension {target_dim}")
            
        # Define the padded observation space
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([self.original_obs_space.low, np.zeros(target_dim - self.original_obs_space.shape[0])]),
            high=np.concatenate([self.original_obs_space.high, np.zeros(target_dim - self.original_obs_space.shape[0])]),
            dtype=np.float32
        )
        # Note: We use zeros for padding in high/low, but really the values will just be zero-filled.
        # SB3 might complain if high < low, so let's set them properly.
        low_pad = np.full(target_dim - self.original_obs_space.shape[0], -np.inf)
        high_pad = np.full(target_dim - self.original_obs_space.shape[0], np.inf)
        
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([self.original_obs_space.low, low_pad]).astype(np.float32),
            high=np.concatenate([self.original_obs_space.high, high_pad]).astype(np.float32),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._pad_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._pad_obs(obs), reward, terminated, truncated, info

    def _pad_obs(self, obs):
        padding = np.zeros(self.target_dim - obs.shape[0])
        return np.concatenate([obs, padding]).astype(np.float32)
