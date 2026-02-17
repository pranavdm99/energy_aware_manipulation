
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.env_factory import make_env

def set_descriptor(env, descriptor):
    """Recursively find LanguageConditionedWrapper and set descriptor."""
    current = env
    while hasattr(current, "env"):
        if hasattr(current, "set_descriptor"):
            current.set_descriptor(descriptor)
            return True
        current = current.env
    
    if hasattr(current, "set_descriptor"):
        current.set_descriptor(descriptor)
        return True
        
    return False

def get_descriptor(env):
    """Recursively find LanguageConditionedWrapper and get descriptor."""
    current = env
    while hasattr(current, "env"):
        if hasattr(current, "descriptor"):
            return current.descriptor
        current = current.env
    
    if hasattr(current, "descriptor"):
        return current.descriptor
        
    return None

def main():
    print("Creating environment...")
    env = make_env(task="Lift", language_conditioned=True, descriptor="normally")
    
    initial_desc = get_descriptor(env)
    print(f"Initial descriptor: {initial_desc}")
    assert initial_desc == "normally"
    
    print("Setting descriptor to 'gently'...")
    success = set_descriptor(env, "gently")
    print(f"Set success: {success}")
    
    new_desc = get_descriptor(env)
    print(f"New descriptor: {new_desc}")
    assert new_desc == "gently"
    
    print("Setting descriptor to 'quickly'...")
    set_descriptor(env, "quickly")
    print(f"New descriptor: {get_descriptor(env)}")
    assert get_descriptor(env) == "quickly"
    
    print("Verification Passed!")

if __name__ == "__main__":
    main()
