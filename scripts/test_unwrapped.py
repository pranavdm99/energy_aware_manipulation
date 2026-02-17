
import robosuite as suite
from robosuite.wrappers import GymWrapper
import gymnasium as gym

def inspect_env(task):
    print(f"\n--- {task} ---")
    env = suite.make(
        env_name=task,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    gym_env = GymWrapper(env)
    
    # Access base env
    base_env = gym_env.unwrapped
    print(f"Base Env Type: {type(base_env)}")
    
    if task == "Lift":
        print(f"Has 'cube': {hasattr(base_env, 'cube')}")
        if hasattr(base_env, 'cube'):
            print(f"Cube ID: {base_env.cube.root_body}")
            
    elif task == "PickPlace":
        print(f"Has 'objects': {hasattr(base_env, 'objects')}")
        if hasattr(base_env, 'objects'):
            print(f"Objects: {[obj.name for obj in base_env.objects]}")
        print(f"Has 'object_id': {hasattr(base_env, 'object_id')}")
        
    elif task == "Door":
         print(f"Has 'door': {hasattr(base_env, 'door')}")
         
    elif task == "NutAssembly":
         print(f"Has 'nut': {hasattr(base_env, 'nut')}")
         print(f"Has 'nuts': {hasattr(base_env, 'nuts')}")

inspect_env("Lift")
inspect_env("PickPlace")
inspect_env("Door")
inspect_env("NutAssembly")
