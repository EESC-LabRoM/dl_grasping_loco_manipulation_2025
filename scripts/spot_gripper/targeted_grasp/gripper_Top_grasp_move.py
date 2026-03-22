import numpy as np
import genesis as gs

# init 
gs.init(backend=gs.cuda, precision='32', theme='dark', eps=1e-12)

# create a scene 
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30, max_FPS=600),
    sim_options=gs.options.SimOptions(dt=0.001),
    show_viewer=True,
    show_FPS=False,
)

# entities 
plane = scene.add_entity(gs.morphs.Plane())
cylinder_radius = 0.03
cylinder_pos = [0.0, 0.0, 0.05] 

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
        file='/home/nexus/Desktop/Genesis/genesis/assets/urdf/spot_arm/urdf/open_gripper.urdf',
        euler=(0, 90, 0),
        pos=(0, 0.0, 0.30),
        scale=1.0,
        merge_fixed_links=True,
        # fixed=True removed to allow movement
    ),
)

# camera setup
# cam = scene.add_camera(
#     pos=(3, -1, 1.5),
#     lookat=(0.0, 0.0, 0.5),
#     res=(1280, 960),
#     fov=30,
#     GUI=False
# )How 

# build 
scene.build()
# cam.start_recording()

print(spot_gripper)
print(spot_gripper.n_dofs)  # Should be 8 (6 for free base + 2 revolute)

# Define DOFs
hand_dof = np.arange(8)  # All 8 DOFs for movement
finger_dof = np.array([7])  # Finger DOF (adjusted for new DOF count)

# Set PD control parameters for all 8 DOFs
spot_gripper.set_dofs_kp(np.array([100, 100, 100, 100, 100, 100, 100, 100]))
spot_gripper.set_dofs_kv(np.array([1, 1, 1, 1, 1, 1, 1, 1]))
spot_gripper.set_dofs_force_range(
    np.array([-100]*8),
    np.array([100]*8))

# Step 1: Grasp the cylinder
print("\033[1;92mGrasping...\033[0m")
current_qpos = spot_gripper.get_dofs_position()
print("Initial DOF positions:", current_qpos)

# Parameters for gripper smooth motion
finger_length = 0.09
start_gripper_pos = current_qpos[finger_dof].item()  # Should be 0 (open)
end_gripper_pos = (cylinder_radius / finger_length) * np.pi
num_steps = 5
pause_steps = 10
gripper_step_size = (end_gripper_pos - start_gripper_pos) / num_steps  # e.g., (-1.571 - 0) / 5 = -0.314

print("Executing paused grasp...")
for i in range(num_steps + 1):
    current_gripper_pos = start_gripper_pos + i * gripper_step_size
    target_qpos = current_qpos.clone()
    target_qpos[finger_dof] = current_gripper_pos
    spot_gripper.control_dofs_position(target_qpos)
    print(f"Step {i}: Gripper at {current_gripper_pos:.3f}")
    for _ in range(pause_steps):
        scene.step()
        # if i % 5:  # Adjusted for smoother recording
        #     cam.render()
current_qpos = spot_gripper.get_dofs_position()
print(current_qpos[finger_dof])
print("Fixing finger position...")
spot_gripper.set_dofs_position( current_qpos[finger_dof], finger_dof)# Step 2: Stabilize the grasp
print("Stabilizing grasp...")
for i in range(150):
    scene.step()
    # if i % 5:
    #     cam.render()

# Step 3: Move the gripper with the cylinder
print("Moving with grasped cylinder...")
linear_velocity = 1.0  # Move right at 0.5 m/s
move_steps = 1000  # Move for 1 second (1000 * 0.001s)

for i in range(move_steps):
    # Maintain grasp by continuing to command closed position
    # target_qpos = spot_gripper.get_dofs_position().clone()
    # target_qpos[finger_dof] = -np.pi/2  # Keep closed
    # spot_gripper.control_dofs_position(target_qpos)
    
    # Set X velocity to move
    velocity = np.array([0, 0, linear_velocity, 0, 0, 0, 0, 0])
    spot_gripper.set_dofs_velocity(velocity, hand_dof)
    
    scene.step()
    # if i % 5:
    #     cam.render()

# Final stabilization
print("Final stabilization...")
for i in range(150):
    scene.step()
#     if i % 5:
#         cam.render()

# cam.stop_recording(save_to_filename="Gripper_Moving_After_Top_Grasp_video.mp4")