
import robosuite as suite
from robosuite.wrappers import GymWrapper

def check_body_id(task):
    print(f"\n--- {task} ---")
    env = suite.make(task, robots="Panda", has_renderer=False)
    gym_env = GymWrapper(env)
    unwrapped = gym_env.unwrapped
    sim = unwrapped.sim
    
    # Try to find obj_body_id
    if hasattr(unwrapped, "obj_body_id"):
        bid = unwrapped.obj_body_id
        # obj_body_id might be a dict (e.g. for multiple objects) or int
        print(f"obj_body_id type: {type(bid)}")
        print(f"obj_body_id value: {bid}")
        
        if isinstance(bid, dict):
            for name, id_val in bid.items():
                print(f"  {name}: {sim.model.body_id2name(id_val)}")
        elif isinstance(bid, int):
             print(f"  Body Name: {sim.model.body_id2name(bid)}")
        # Check specific attributes if obj_body_id is missing/weird
    else:
        print("No obj_body_id attribute.")
        # Fallback checks
        if hasattr(unwrapped, "cube_body_id"):
             print(f"cube_body_id: {sim.model.body_id2name(unwrapped.cube_body_id)}")

check_body_id("Lift")
check_body_id("PickPlace")
check_body_id("NutAssembly")
check_body_id("Door")
