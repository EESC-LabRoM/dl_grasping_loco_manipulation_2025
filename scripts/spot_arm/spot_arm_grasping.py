import genesis as gs
import numpy as np

# init genesis
gs.init(
    backend             = gs.cuda,
    seed                = None,
    precision           = '32',
    debug               = False,
    eps                 = 1e-16,
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
        max_FPS       = 600,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
        gravity=(0, 0, -10.0),
    ),
    show_viewer = True,
    show_FPS=False,
    show_FPS=False,
)

# entities
plane = scene.add_entity(gs.morphs.Plane())
spot_arm = scene.add_entity(
    gs.morphs.URDF(
        file  = '/home/nexus/Desktop/Genesis/genesis/assets/urdf/spot_arm/urdf/spot_arm_description.urdf',
        euler = (0, 0, 30), 
        scale = 1.0,
        pos   = (0, 0, 0.1),
        merge_fixed_links=False,
        convexify=True,
        fixed=True
    ),
)
cylinder = scene.add_entity(
    gs.morphs.Cylinder(
        pos = (0.65, 0.2, 0.02),
        height = 0.1,
        radius = 0.03,
        collision = True,
    )
)
# camera setup
# cam = scene.add_camera(
#     pos    = (3, -1, 1.5),
#     lookat = (0.0, 0.0, 0.5),
#     res    = (1280, 960),
#     fov    = 30,
#     GUI    = False
# )

#build
scene.build()
cylinder.set_pos( np.array([0.65, 0.2, 0.02]), zero_velocity=True)

print(spot_arm)
print(spot_arm.n_dofs)

# cam.start_recording()

motors_dof = np.arange(5)
finger_dof = np.array([spot_arm.n_dofs - 1]) 

print("Motor DOFs total: ", spot_arm.n_dofs)
print("Gripper DOFs: ", finger_dof)

# PD Control parameters 
spot_arm.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 1000]),
)
spot_arm.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200]),
)
spot_arm.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -10]),
    np.array([ 87,  87,  87,  87,  12,  10]),
)

# get the end-effector link
end_effector = spot_arm.get_link('arm_link_fngr')

# move to pre-grasp pose
print("Moving to pre-grasp the cylinder...")
target_links_position = spot_arm.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.60, 0.2, 0.35]),
    quat = np.array([-0.7, 0, -0.7, 0]),
)

# gripper open pos
target_links_position[-1:] = - np.pi/2

path = spot_arm.plan_path(
    qpos_goal     = target_links_position,
    num_waypoints = 1000, # 2s duration
)
print("Executing planned position...")
for n, waypoint in enumerate(path):
    spot_arm.control_dofs_position(waypoint)
    scene.step()
    # if n%10:
    #     cam.render()

# allow robot to reach the last waypoint
for i in range(1000):
    scene.step()
    # if i%10:
    #     cam.render()

# reach
print("Reaching...")
target_links_position = spot_arm.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.51, 0.2, 0.2]),
    quat = np.array([0.7, -0.7, 0, 0]),
    damping = 0.1,

)

spot_arm.control_dofs_position(target_links_position[:-1], motors_dof)
for i in range(1, 1000):
    scene.step()
    # cam.render()

# reach
print("Reaching even closer...")
target_links_position = spot_arm.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.59, 0.25, 0.1]),
    quat = np.array([0.7, -0.7, 0, 0.7]),
    damping = 0.1,

)

spot_arm.control_dofs_position(target_links_position[:-1], motors_dof)
for i in range(1000):
    scene.step()
    # cam.render()

# grasp
print("Grasping...")

# Turn 
print("Gripper going to 0 position...")
# spot_arm.control_dofs_position(np.array([0]), finger_dof) ~it is really much fast
current_qpos = spot_arm.get_dofs_position()

# Parameters for gripper smooth motion
start_gripper_pos = -np.pi/2  
end_gripper_pos = -np.pi/5            
num_steps = 5                 
pause_steps = 50              

# Calculate step size for gripper DOF
gripper_step_size = (end_gripper_pos - start_gripper_pos) / num_steps

# Execute the grasping motion in paused increments 
print("Executing paused grasp...")
for i in range(num_steps + 1):  
    # Calculate the current gripper position 
    current_gripper_pos = start_gripper_pos + i * gripper_step_size

    # Update only the gripper DOF ~Genesis RigidEntity's API is Satana's made!
    target_qpos = current_qpos.clone()
    target_qpos[finger_dof] = current_gripper_pos
    spot_arm.control_dofs_position(target_qpos)
    
    # Pause for a duration ~I tried use the plan_path but I got various bugs...
    print(f"Step {i}: Gripper at {current_gripper_pos:.3f}")
    for _ in range(pause_steps):
        scene.step()
        # if i % 10 == 0:
        #     cam.render()
for i in range(1, 1000):
    scene.step()
    # cam.render()

# lift
print("Lifting...")
qpos = spot_arm.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.2, 0.28]),
    quat = np.array([0.7, 0, 0, 0]),

)
spot_arm.control_dofs_position(qpos[:-1], motors_dof)
for i in range(200):
    scene.step()
#     cam.render()

# cam.stop_recording(save_to_filename="Spot_Arm_Grasping_video.mp4")