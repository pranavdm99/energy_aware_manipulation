
import robosuite as suite
import numpy as np

TASKS = ["Door", "PickPlace", "NutAssembly"]

for task in TASKS:
    print(f"\nScanning task: {task}")
    try:
        env = suite.make(
            env_name=task,
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )
        env.reset()
        sim = env.sim
        
        print(f"  Body Names: {sim.model.body_names}")
        print(f"  Site Names: {sim.model.site_names}")
        
    except Exception as e:
        print(f"  Error loading {task}: {e}")
