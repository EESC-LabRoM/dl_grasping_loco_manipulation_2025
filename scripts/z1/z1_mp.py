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
    pos    = (3.5, 2.0, 2.5),
    lookat = (0, 0, 0.5),
    fov    = 30,
    GUI    = False
)

# build 
scene.build()
print(z1)
print(z1.n_dofs)
cam.start_recording()

motors_dof = np.arange(6)
fingers_dof = np.arange(7)

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
end_effector = z1.get_link('link06')
# end_effector = z1.get_link('gripperMover')

# move to pre-grasp pose
qpos = z1.inverse_kinematics(
    link = end_effector,
    pos  = np.array([0.65, 0.0, 0.35]),
    # quat = np.array([0, 1, 0, 0]),
    quat = np.array([0.7, -0.7, 0, 0]),

)
# gripper open pos
qpos[-1:] = 0.06
path = z1.plan_path(
    qpos_goal     = qpos,
    num_waypoints = 200, # 2s duration
)
# execute the planned path
for waypoint in path:
    z1.control_dofs_position(waypoint)
    scene.step()
    cam.set_pose(
    pos = (3.5, 2.0, 2.5),
    lookat = (0, 0, 0.5),
    )
    cam.render()
cam.stop_recording(save_to_filename='z1_mp_video.mp4', fps=60)
