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
cylinder = scene.add_entity(
    gs.morphs.Cylinder(
        pos = (0.65, 0.0, 0.02),
        height = 0.1,
        radius = 0.03,
        collision = True,
    )
)

z1 = scene.add_entity(
    gs.morphs.URDF(
        file  = '/home/nexus/Desktop/Genesis/genesis/assets/urdf/z1/xacro/z1.urdf',
        euler = (0, 0, 45),
        scale = 1.0,
        merge_fixed_links=True,
        fixed=True
    ),
)

#camera setup
cam = scene.add_camera(
    res    = (1280, 960),
    pos    = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = False
)

# build 
scene.build()
cam.start_recording()

print(z1)
motors_dof = np.arange(6)
#Z1 only has 1 motor finger, which is the last joint of the tensor.
fingers_dof = np.array([z1.n_dofs - 1]) 

print("Total DOFs:", z1.n_dofs)
print("DOF da garra:", fingers_dof)

z1.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 100]),
)
z1.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200]),
)
z1.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, 10]),
    np.array([ 87,  87,  87,  87,  12,  12,  -10]),
)
# get the end-effector link
end_effector = z1.get_link('gripperMover')

# move to pre-grasp pose
print("Moving to pre-grasp the cylinder...")
qpos = z1.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.35]),
    quat = np.array([0.7, -0.7, 0, 0]),
)
# gripper open pos
print("Planning Opening Gripper position...")
qpos[-1:] = 0.03
path = z1.plan_path(
    qpos_goal     = qpos,
    num_waypoints = 200, # 2s duration
)
print(path)
# execute the planned path
print("Executing planned position...")
for waypoint in path:
    z1.control_dofs_position(waypoint)
    scene.step()
    cam.set_pose(
    pos = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    )
    cam.render()
# allow robot to reach the last waypoint
for i in range(100):
    scene.step()
    cam.set_pose(
    pos = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    )
    cam.render()

# reach
print("Reaching...")
qpos = z1.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.100]),
    quat = np.array([0.7, -0.7, 0, 0]),
)

z1.control_dofs_position(qpos[:-1], motors_dof)
for i in range(100):
    scene.step()
    cam.set_pose(
    pos = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    )
    cam.render()
# grasp
print("Grasping...")
z1.control_dofs_position(qpos[:-1], motors_dof)

for i in range(100):
    z1.set_dofs_position(np.array([0.01]), fingers_dof)
    scene.step()
    cam.set_pose(
    pos = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    )
    cam.render()

# lift
print("Grasping...")
qpos = z1.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.28]),
    quat = np.array([0.7, -0.7, 0, 0]),

)
z1.control_dofs_position(qpos[:-1], motors_dof)
for i in range(200):
    scene.step()
    cam.set_pose(
    pos = (3.5, 0.0, 2.5),
    lookat = (0, 0, 0.5),
    )
    cam.render()
cam.stop_recording(save_to_filename='Z1_Grasping_video.mp4', fps=60)
