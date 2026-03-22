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

wine_bottle = scene.add_entity(
    gs.morphs.Mesh(
        file="/home/nexus/Desktop/Genesis/examples/Exp_RigidEntity/bottles/wine-bottle/bottle.obj",  # Absolute or relative path
        pos=(-0.5, 0.0, 0.1),            
        euler=(90, 0, 0.0),
        collision=True,
        visualization=True,
        fixed=True,          
        scale=0.001                      
    )
)

water_bottle = scene.add_entity(
    gs.morphs.Mesh(
        file="/home/nexus/Desktop/Genesis/examples/Exp_RigidEntity/bottles/bottled-water/Bottle_v1.stl",  # Absolute or relative path
        pos=(0.5, 0.0, 0),            
        euler=(0, 0, 0.0),
        collision=True,
        visualization=True,
        fixed=True,          
        scale=1                      
    )
)

beer_bottle = scene.add_entity(
    gs.morphs.Mesh(
        file="/home/nexus/Desktop/Genesis/examples/Exp_RigidEntity/bottles/beer_bottle//beer_bottle.obj",  # Absolute or relative path
        pos=(0.0, 0.0, 0),            
        euler=(0, 0, 0.0),
        collision=True,
        visualization=True,
        fixed=True,          
        scale=0.01                      
    )
)

thermo_bottle = scene.add_entity(
    gs.morphs.Mesh(
        file="/home/nexus/Desktop/Genesis/examples/Exp_RigidEntity/bottles/thermo-bottle/thermo_bottle.glb",  # Absolute or relative path
        pos=(0.0, -0.5, 0.065),            
        euler=(90, 0, 0.0),
        collision=True,
        visualization=True,
        fixed=True,          
        scale=0.0001                      
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


# cam.start_recording()

for i in range(120):
    scene.step()
    cam.render()
input("Enter")
    
# cam.stop_recording(save_to_filename='cylinder_video.mp4', fps=60)