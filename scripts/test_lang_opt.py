
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.env_factory import create_vec_env

def test_language_optimization():
    config = {
        "environment": {
            "task": "Lift",
            "robots": "Panda",
            "controller": "OSC_POSE",
            "horizon": 100,
        },
        "energy": {
            "weight": 0.0,
        },
        "language": {
            "enabled": True,
            "descriptor": "gently",
            "model": "all-MiniLM-L6-v2",
        }
    }
    
    print("Creating vectorized environment (2 envs)...")
    # This should print "Pre-calculating language embeddings..." exactly once
    vec_env = create_vec_env(config, n_envs=2, seed=42)
    
    print(f"Observation space: {vec_env.observation_space}")
    
    # Check if embeddings were added to config
    if "pre_calculated_embeddings" in config["language"]:
        print("SUCCESS: Pre-calculated embeddings found in config.")
        emb = config["language"]["pre_calculated_embeddings"]["gently"]
        print(f"Embedding shape: {emb.shape}")
    else:
        print("FAILURE: Pre-calculated embeddings NOT found in config.")
        
    obs = vec_env.reset()
    print(f"Observation batch shape: {obs.shape}")
    
    vec_env.close()

if __name__ == "__main__":
    test_language_optimization()
