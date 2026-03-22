import numpy as np
import genesis as gs

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
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5), 
        camera_lookat=(0.0, 0.0, 0.5), 
        camera_fov=30, max_FPS=600
        ),
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
        euler=(90, 0, 0),
        pos=(-0.2, 0.0, 0.10),
        scale=1.0,
        merge_fixed_links=True,
        # fixed=True removed to allow movement
    ),
)

# build 
scene.build()

print(spot_gripper)
print(spot_gripper.n_dofs)  # Should be 8

# Define DOFs
hand_dof = np.arange(8)
finger_dof = np.array([7])

# Set PD control parameters
spot_gripper.set_dofs_kp(np.array([100, 100, 100, 100, 100, 100, 100, 100]))
spot_gripper.set_dofs_kv(np.array([1, 1, 1, 1, 1, 1, 1, 1]))
spot_gripper.set_dofs_force_range(np.array([-100]*8), np.array([100]*8))

# Grasp parameters
finger_length = 0.09
num_steps = 5
pause_steps = 10

#Linear motion parameters
linear_velocity = 1.0  
move_steps = 500  

# Function to reset cylinder position
def reset_cylinder():
    for i in range(150):
        scene.step()
    cylinder.set_pos(np.array(cylinder_pos), zero_velocity=True)
    cylinder.set_quat(np.array([0.7, 0, 0, 0.7]), zero_velocity=True)

# Function to perform Linear Motion
def linear_motion():
    # Move the gripper with the cylinder
    print("\033[1;92mMoving with grasped cylinder...\033[0m")
    for i in range(move_steps):

    # Set X velocity to move
        velocity = np.array([0, 0, linear_velocity, 0, 0, 0, 0, 0])
        spot_gripper.set_dofs_velocity(velocity, hand_dof)
        scene.step()
    # Final stabilization
    print("Final stabilization...")
    for i in range(150):
        scene.step()
    

# Function to perform grasp at current position
def perform_grasp():
    print("\033[1;92mGrasping...\033[0m")
    current_qpos = spot_gripper.get_dofs_position()
    start_gripper_pos = current_qpos[finger_dof].item()  # Should be 0 (open)
    end_gripper_pos = (cylinder_radius / finger_length) * np.pi # Closed position
    gripper_step_size = (end_gripper_pos - start_gripper_pos) / num_steps

    print("\033[1;92mExecuting paused grasp...\033[0m")
    for i in range(num_steps + 1):
        current_gripper_pos = start_gripper_pos + i * gripper_step_size
        target_qpos = current_qpos.clone()
        target_qpos[finger_dof] = current_gripper_pos
        spot_gripper.control_dofs_position(target_qpos)
        print(f"Step {i}: Gripper at {current_gripper_pos:.3f}")
        for _ in range(pause_steps):
            scene.step()

    print("Stabilizing grasp...")
    for i in range(150):
        scene.step()
    # Perform the Linear Motion
    linear_motion()
    spot_gripper.control_dofs_position(current_qpos)
    for i in range(150):
        scene.step()

# Test grasp at initial position [0, 0, 0.3]
print("Testing grasp at initial position \033[1;94m[0, 0, 0.3]\033[0m")
perform_grasp()

# Reset cylinder and gripper for next test
reset_cylinder()
spot_gripper.set_pos(np.array([-0.2, 0.0, 0.10]), zero_velocity=True)

# Test grasp at varying Y positions
print("\033[1;92mTesting Y translate position grasping\033[0m")
y_positions = np.arange(-cylinder_radius, cylinder_radius + 0.01, 0.01)
for y in y_positions:
    print(f"\nTesting grasp at position \033[1;94m[0, {y:.2f}, 0.3]\033[0m")
    # Move gripper to new Y position
    spot_gripper.set_pos(np.array([-0.2, y, 0.10]), zero_velocity=True)
    for _ in range(50):  # Stabilize position
        scene.step()
    
    # Perform grasp
    perform_grasp()
    
    # Reset for next iteration
    reset_cylinder()
    spot_gripper.set_pos(np.array([-0.2, 0.0, 0.10]), zero_velocity=True)
    for _ in range(50):
        scene.step()

print("\033[1;92mTesting complete.\033[0m")