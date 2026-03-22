import numpy as np
import genesis as gs

# init 
gs.init(backend=gs.cuda, precision='32', theme='dark', eps=1e-12)

# create a scene 
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30, max_FPS=600),
    sim_options=gs.options.SimOptions(dt=0.001),
    show_viewer=True,
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
        pos=(0, 0, 0.3),
        scale=1.0,
        merge_fixed_links=True,
        fixed = True
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
hand_dof = np.arange(2)
finger_dof = np.array([1]) 

# Set PD control parameters
spot_gripper.set_dofs_kp(
    np.array([100, 100])
    )
spot_gripper.set_dofs_kv(
    np.array([ 1, 1])
    )
spot_gripper.set_dofs_force_range(
    np.array([ -100, -100]),
    np.array([ 100, 100])
    )

print("Grasping...")
current_qpos = spot_gripper.get_dofs_position()

# Parameters for gripper smooth motion
finger_length = 0.1
start_gripper_pos = current_qpos[finger_dof].item()
end_gripper_pos = (cylinder_radius / finger_length) * np.pi 
num_steps = 5
pause_steps = 10
gripper_step_size = (end_gripper_pos - start_gripper_pos) / num_steps  # e.g., (0 - (-1.571)) / 5 = 0.314
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

cam.stop_recording(save_to_filename="Gripper_Static_Top_Grasping_video.mp4")