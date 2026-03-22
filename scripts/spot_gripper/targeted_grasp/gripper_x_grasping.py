import numpy as np
import genesis as gs

# init 
gs.init(backend=gs.cuda, precision='32', theme='dark', eps=1e-12)

# create a scene 
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(camera_pos=(3, -1, 1.5), camera_lookat=(0.0, 0.0, 0.5), camera_fov=30, max_FPS=60),
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=True,
)

# entities 
plane = scene.add_entity(gs.morphs.Plane())
cylinder_radius = 0.03
cylinder_pos = [0.0, 0.0, 0.02] 

cylinder = scene.add_entity(
    gs.morphs.Cylinder(
        pos=cylinder_pos,
        height=0.1,
        radius=cylinder_radius,
        collision=True,
    )
)

spot_gripper = scene.add_entity(
    gs.morphs.URDF(
        file='/home/nexus/Desktop/Genesis/genesis/assets/urdf/spot_arm/urdf/gripper.urdf',
        euler=(90, 0, 0),
        pos=(-0.6, 0.0, 0.10),
        scale=1.0,
        merge_fixed_links=True,
    ),
)
# camera setup
cam = scene.add_camera(
    pos    = (3, -1, 1.5),
    lookat = (0.0, 0.0, 0.5),
    res    = (1280, 960),
    fov    = 30,
    GUI    = False
)

# build 
scene.build()
cam.start_recording()

print(spot_gripper)
print(spot_gripper.n_dofs)

# Define DOFs
hand_dof = np.arange(8)
finger_dof = np.array([7]) 

# Set PD control parameters
spot_gripper.set_dofs_kp(
    np.array([100, 100, 100, 0, 0, 0, 100, 100])
    )
spot_gripper.set_dofs_kv(
    np.array([1, 1, 1, 0, 0, 0, 1, 1])
    )
spot_gripper.set_dofs_force_range(
    np.array([-10, -10, -10, 0, 0, 0, -100, -100]),
    np.array([10, 10, 10, 0, 0, 0, 100, 100])
    )

# Step 1: Open the gripper fingers to -np.pi/2
print("Opening gripper fingers...")
current_qpos = spot_gripper.get_dofs_position()
target_qpos = current_qpos.clone()
target_qpos[finger_dof] = -np.pi/2  # Fully open
spot_gripper.control_dofs_position(target_qpos)

# Let the gripper settle in open position
for i in range(100):  # 1 second to stabilize   
    scene.step()
    if i%10:
        cam.render() 
# Step 2: Move gripper toward cylinder
print("Moving toward cylinder...")
base_link = spot_gripper.get_link('arm_link_wr1')
linear_velocity = 0.5  # Speed in m/s
break_distance = 0.2  # Stop when within 'n' meters
max_steps = 1000  # Safety limit

for i in range(max_steps):
    base_pos = base_link.get_pos()
    linear_distance = cylinder_pos[0] - base_pos[0]  # X-linear-difference

    print(f"Step {i}: Wrist X={base_pos[0]:.3f}, Distance to cylinder={linear_distance:.3f}")

    if abs(linear_distance) < break_distance:
        print("Close enough to cylinder, stopping movement.")
        spot_gripper.set_dofs_velocity(np.array([0.0, 0, 0, 0, 0, 0, 0, 0]), hand_dof)  # Stop
        break
    else:
        velocity = np.array([linear_velocity if linear_distance > 0 else -linear_velocity, 0, 0, 0, 0, 0, 0, 0])
        spot_gripper.set_dofs_velocity(velocity, hand_dof)
    
    scene.step()
    if i%10:
        cam.render()

# Step 3: Grasping phase
print("Grasping...")
current_qpos = spot_gripper.get_dofs_position()

# Parameters for gripper smooth motion
finger_length = 0.1
start_gripper_pos = -np.pi/2  # Already open from Step 1
end_gripper_pos = (cylinder_radius / finger_length) * np.pi + start_gripper_pos
num_steps = 5
pause_steps = 100
gripper_step_size = (end_gripper_pos - start_gripper_pos) / num_steps

print("Executing paused grasp...")
for i in range(num_steps + 1):
    current_gripper_pos = start_gripper_pos + i * gripper_step_size
    target_qpos = current_qpos.clone()
    target_qpos[finger_dof] = current_gripper_pos
    spot_gripper.control_dofs_position(target_qpos)
    print(f"Step {i}: Gripper at {current_gripper_pos:.3f}")
    for _ in range(pause_steps):
        scene.step()
        if i%10:
            cam.render()

# Stabilize
print("Stabilizing...")
for i in range(150):
    scene.step()
    if i%10:
        cam.render()

cam.stop_recording(save_to_filename="Gripper_X_Grasping_video.mp4")