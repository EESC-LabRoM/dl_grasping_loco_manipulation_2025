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
        dt = 0.001,
        #gravity=(0, 0, -10.0),
    ),
    show_viewer = True,
    show_FPS=False,
)

# entities
plane = scene.add_entity(gs.morphs.Plane())
spot_arm = scene.add_entity(
    gs.morphs.URDF(
        file  = '/home/nexus/Desktop/Genesis/genesis/assets/urdf/spot_arm/urdf/spot_arm_description.urdf',
        euler = (0, 0, 0), 
        scale = 1.0,
        pos   = (0, 0, 0.1),
        merge_fixed_links=False,
        convexify=True,
        fixed=True
    ),
)
cylinder = scene.add_entity(
    gs.morphs.Cylinder(
        pos = (0.65, 0.0, 0.02),
        height = 0.1,
        radius = 0.03,
        collision = True,
        # fixed = True,
    )
)
#camera setup
cam = scene.add_camera(
    pos    = (3, -1, 1.5),
    lookat = (0.0, 0.0, 0.5),
    res    = (1280, 960),
    fov    = 30,
    GUI    = False
)

#build
scene.build()
cylinder.set_pos( np.array([0.65, 0.0, 0.02]), zero_velocity=True)   
print(spot_arm)
print(spot_arm.n_dofs)
cam.start_recording()

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

# # get the end-effector link
end_effector = spot_arm.get_link('arm_link_fngr')

# qpos = spot_arm.inverse_kinematics(
#     link = end_effector,
#     quat = np.array([-0.7, 0.7, 0, 0]),
# )

# move to pre-grasp pose
print("Moving to pre-grasp the cylinder...")
target_links_position = spot_arm.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.60, 0.0, 0.35]),
    quat = np.array([-0.7, 0.7, 0, 0]),
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
    if n%10:
        cam.render()

# input("Click enter...")

# allow robot to reach the last waypoint
for i in range(1000):
    scene.step()
    if i%10:
        cam.render()

# input("Click enter...")
cylinder.set_pos( np.array([0.65, 0.0, 0.02]), zero_velocity=True)

# reach
print("Reaching...")
target_links_position = spot_arm.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.51, 0.0, 0.00]),
    quat = np.array([-0.7, 0.7, 0, 0]),
)

spot_arm.control_dofs_position(target_links_position[:-1], motors_dof)
for i in range(100):
    scene.step()
    cam.render()
# input("Click enter...")

# reach
print("Reaching even closer...")
target_links_position = spot_arm.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.56, 0.0, 0.060]),
    quat = np.array([-0.7, 0.7, 0, 0]),
)

spot_arm.control_dofs_position(target_links_position[:-1], motors_dof)
for i in range(1000):
    scene.step()
    cam.render()
# input("Click enter...")

# grasp
print("Grasping...")
# spot_arm.control_dofs_position(qpos[:-1], motors_dof)
#qpos[-1:] =  np.pi/2
#spot_arm.control_dofs_position(qpos[-1:], finger_dof)

print("0")
spot_arm.control_dofs_position(np.array([0]), finger_dof)
for i in range(1, 1000):
    scene.step()
    cam.render()

# input("Click enter...")

# print("pi/2")
# spot_arm.control_dofs_position(np.array([np.pi/2]), finger_dof)
# for i in range(1, 100):
#     scene.step()

# input("Click enter...")


# lift
print("Lifting...")
qpos = spot_arm.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.28]),
    quat = np.array([0.7, -0.7, 0, 0]),

)
spot_arm.control_dofs_position(qpos[:-1], motors_dof)
for i in range(200):
    scene.step()
    cam.render()

cam.stop_recording(save_to_filename="video.mp4")