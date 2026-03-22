import numpy as np

import genesis as gs

########################## init ##########################
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

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        res           = (960, 640),
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    show_viewer = True,
)

########################## entities ##########################
plane = scene.add_entity(gs.morphs.Plane())
z1 = scene.add_entity(
    gs.morphs.URDF(
        file  = '/home/nexus/Desktop/Genesis/genesis/assets/urdf/z1/xacro/z1.urdf',
        euler = (0, 0, 90),
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

#################### build ##########################
scene.build()
print(z1)
print(z1.n_dofs)
cam.start_recording()

jnt_names = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'jointGripper',
    'gripperStator',
]
dofs_idx = [z1.get_joint(name).dof_idx_local for name in jnt_names]

############ Optional: set control gains ############
# set positional gains
z1.set_dofs_kp(
    kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 100, 100]),
    dofs_idx_local = dofs_idx,
)
# set velocity gains
z1.set_dofs_kv(
    kv             = np.array([450, 450, 350, 350, 200, 200, 10, 10]),
    dofs_idx_local = dofs_idx,
)
# set force range for safety
z1.set_dofs_force_range(
    lower          = np.array([-87, -87, -87, -87, -12, -12, -100, -100]),
    upper          = np.array([ 87,  87,  87,  87,  12,  12, 100, 100]),
    dofs_idx_local = dofs_idx,
)
# Hard reset
for i in range(150):
    if i < 50:
        z1.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
    elif i < 100:
        z1.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, 0.04, 0.04]), dofs_idx)
    else:
        z1.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0]), dofs_idx)

    scene.step()

# PD control
for i in range(1250):
    if i == 0:
        z1.control_dofs_position(
            np.array([1, 1, 0, 0, 0, 0, 0.04, 0.04]),
            dofs_idx,
        )
    elif i == 250:
        z1.control_dofs_position(
            np.array([-1, 0.8, 1, -2, 1, 0.5, 0.04, 0.04]),
            dofs_idx,
        )
    elif i == 500:
        z1.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
    elif i == 750:
        # control first dof with velocity, and the rest with position
        z1.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0])[1:],
            dofs_idx[1:],
        )
        z1.control_dofs_velocity(
            np.array([1.0, 0, 0, 0, 0, 0, 0, 0])[:1],
            dofs_idx[:1],
        )
    elif i == 1000:
        z1.control_dofs_force(
            np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )
    # This is the control force computed based on the given control command
    # If using force control, it's the same as the given control command
    print('control force:', z1.get_dofs_control_force(dofs_idx))

    # This is the actual force experienced by the dof
    print('internal force:', z1.get_dofs_force(dofs_idx))

    scene.step()
    cam.set_pose(
    pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
    lookat = (0, 0, 0.5),
    )
    cam.render()
cam.stop_recording(save_to_filename='z1_stator_video.mp4', fps=60)