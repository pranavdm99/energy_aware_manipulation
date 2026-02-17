
import robosuite as suite
from robosuite.wrappers import GymWrapper

def check_active_obj(task):
    print(f"\n--- {task} ---")
    env = suite.make(task, robots="Panda", has_renderer=False)
    gym_env = GymWrapper(env)
    gym_env.reset()
    unwrapped = gym_env.unwrapped
    
    if task == "PickPlace":
        if hasattr(unwrapped, "obj_to_use"):
            print(f"obj_to_use: {unwrapped.obj_to_use}")
        else:
            print("No obj_to_use attribute")
            
        # Check single_object_mode
        if hasattr(unwrapped, "single_object_mode"):
             print(f"single_object_mode: {unwrapped.single_object_mode}")
             
    elif task == "Door":
        attrs = [a for a in dir(unwrapped) if "door" in a or "handle" in a]
        print(f"Door/Handle attributes: {attrs}")
        
    elif task == "NutAssembly":
        if hasattr(unwrapped, "nut_to_id"):
             print(f"nut_to_id: {unwrapped.nut_to_id}")
             # In NutAssembly, usually all nuts are targets? Or just one?
             # Check if there's a target nut
             pass

check_active_obj("PickPlace")
check_active_obj("Door")
