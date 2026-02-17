
import robosuite as suite
import numpy as np

# Create the environment
env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)
env.reset()

sim = env.sim
print("Body Names:", sim.model.body_names)
print("Site Names:", sim.model.site_names)

# Check specifically for expected names
print(f"Has 'cube_main' body? {'cube_main' in sim.model.body_names}")
print(f"Has 'gripper0_grip_site'? {'gripper0_grip_site' in sim.model.site_names}")
