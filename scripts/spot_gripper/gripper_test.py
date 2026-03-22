import genesis as gs

# init genesis
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
#    sim_options=gs.options.SimOptions(
#        dt=0.01,
#        gravity=(0, 0, -10.0),
#    ),
    show_viewer    = True,
    viewer_options = gs.options.ViewerOptions(
        res           = (1280, 960),
        camera_pos    = (3.5, 0.0, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 40,
        max_FPS       = 60,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = True, # visualize the coordinate frame of `world` at its origin
        world_frame_size = 1.0, # length of the world frame in meter
        show_link_frame  = False, # do not visualize coordinate frames of entity links
        show_cameras     = False, # do not visualize mesh and frustum of the cameras added
        plane_reflection = True, # turn on plane reflection
        ambient_light    = (0.1, 0.1, 0.1), # ambient light setting
    ),
    renderer = gs.renderers.Rasterizer(), # using rasterizer for camera rendering
)

# entities
plane = scene.add_entity(gs.morphs.Plane())
spot_gripper = scene.add_entity(
    gs.morphs.URDF(
        file  = '/home/nexus/Desktop/Genesis/genesis/assets/urdf/spot_arm/urdf/open_gripper.urdf',
        euler = (0, -90, 0),
        scale = 1.0,
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

#build
scene.build()
print(spot_gripper)

cam.start_recording()
import numpy as np

for i in range(400):
    scene.step()
    cam.set_pose(
        pos = (3.5, 2.0, 2.5),
        lookat = (0, 0, 0.5),
    )
    cam.render()
cam.stop_recording(save_to_filename='spot_gripper_test_video.mp4', fps=60)