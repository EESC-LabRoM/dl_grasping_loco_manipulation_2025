import numpy as np
import genesis as gs
from time import sleep
# init 
gs.init(
    backend             = gs.cuda,
    seed                = None,
    precision           = '32',
    debug               = False,
    eps                 = 1e-12,
    logging_level       = None,
    theme               = 'dark',
    logger_verbose_time = False
    )

# create a scene 
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (3, -1, 1.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = True,
)

# entities 
plane = scene.add_entity(
    gs.morphs.Plane(),
)
cylinder_radius = 0.03
cylinder_pos = [0.0, 0.0, 0.02]  

cylinder = scene.add_entity(
    gs.morphs.Cylinder(
        pos = cylinder_pos,
        height = 0.1,
        radius = cylinder_radius,
        collision = True,
        # fixed = True,
    )
)

spot_gripper = scene.add_entity(
    gs.morphs.URDF(
        file  = '/home/nexus/Desktop/Genesis/genesis/assets/urdf/spot_arm/urdf/gripper.urdf',
        euler = (90, 0, 0),
        pos = (-0.6, 0.0, 0.10),
        scale = 1.0,
        merge_fixed_links=True,
        # fixed=True
    ),
)

# build 

scene.build()
# cam.start_recording()

print(spot_gripper)
print(spot_gripper.n_dofs)

hand_dof = np.arange(8)
finger_dof = np.array([7]) 
spot_gripper.set_dofs_kp(
    np.array([ 100, 100, 100, 0, 0, 0, 100, 100]),
)
spot_gripper.set_dofs_kv(
    np.array([ 1, 1, 1, 0, 0, 0, 1, 1]),
)
spot_gripper.set_dofs_force_range(
    lower=np.array([ -10, -10, -10, 0, 0, 0, -100, -100]),
    upper=np.array([ 10, 10, 10, 0, 0, 0, 100, 100]),
)
#get non-finger link from the gripper
base_link = spot_gripper.get_link('arm_link_wr1')
# Move gripper toward cylinder
print("Moving toward cylinder...")
# Speed in m/s (adjustable)
linear_velocity = 0.5  
# Stop when within 'n' meters of cylinder
break_distance = 0.15  
# Safety limit to avoid infinite loop
max_steps = 1000  

for i in range(max_steps):
    # Get current position of 'arm_link_wr1'
    base_pos = base_link.get_pos()
    # X-difference (cylinder at x=0)
    linear_distance = cylinder_pos[0] - base_pos[0]  

    print(f"Step {i}: Wrist X={base_pos[0]:.3f}, Distance to cylinder={linear_distance:.3f}")

    # Check if close enough
    if abs(linear_distance) < break_distance:
        print("Close enough to cylinder, stopping movement.")
        spot_gripper.set_dofs_velocity(np.array([0.0, 0, 0, 0, 0, 0, 0, 0]), hand_dof)  # Stop
        break
    else:
        # Move in X direction toward cylinder
        velocity = np.array([linear_velocity if linear_distance > 0 else -linear_velocity, 0, 0, 0, 0, 0, 0, 0])
        spot_gripper.set_dofs_velocity(velocity, hand_dof)

    scene.step()

print("Grasping...")

# Turn 
print("Gripper going to 0 position...")
# spot_gripper.control_dofs_position(np.array([0]), finger_dof) ~ it is so much fast, dont recomend.
current_qpos = spot_gripper.get_dofs_position()

# Parameters for gripper smooth motion
finger_length = 0.09 # ~ the entire finger is about 0.125
start_gripper_pos = -np.pi/2  
end_gripper_pos = (cylinder_radius / finger_length) * np.pi + start_gripper_pos           
num_steps = 5                 
pause_steps = 100              

# Calculate step size for gripper DOF 
gripper_step_size = (end_gripper_pos - start_gripper_pos) / num_steps

# Execute the grasping motion in paused increments ~ Genesis RigidEntity's API is Satana's made!
print("Executing paused grasp...")
for i in range(num_steps + 1):  
    # Calculate the current gripper position 
    current_gripper_pos = start_gripper_pos + i * gripper_step_size

    # Update only the gripper DOF 
    target_qpos = current_qpos.clone()
    target_qpos[finger_dof] = current_gripper_pos
    spot_gripper.control_dofs_position(target_qpos)
    
    # Pause for a duration ~I tried use the plan_path but I got various bugs...
    print(f"Step {i}: Gripper at {current_gripper_pos:.3f}")
    for _ in range(pause_steps):
        scene.step()
        # if i % 10 == 0:
        #     cam.render()
for i in range(300):

    scene.step()
