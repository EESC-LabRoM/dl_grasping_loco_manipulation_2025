import genesis as gs
import numpy as np

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
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5), 
        camera_lookat=(0.0, 0.0, 0.5), 
        camera_fov=30, 
        max_FPS=600
        ),
    sim_options=gs.options.SimOptions(
        dt=0.001
        ),
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
        euler=[0,90,0],
    )
)

#camera setup
cam = scene.add_camera(
    pos=(3, -1, 1.5),
    lookat=(0.0, 0.0, 0.5),
    res=(1280, 960),
    fov=30,
    GUI=False
)

#build
scene.build()

# render rgb, depth, segmentation, and normal
rgb, depth, segmentation, normal = cam.render(
    rgb=True, 
    depth=True, 
    segmentation=False, 
    normal=False
    )


cam.start_recording()

for i in range(120):
    scene.step()
    cam.render()
    
cam.stop_recording(save_to_filename='cylinder_video.mp4', fps=60)