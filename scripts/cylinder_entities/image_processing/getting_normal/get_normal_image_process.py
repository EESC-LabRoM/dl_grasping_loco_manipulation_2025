import genesis as gs
import numpy as np

# init genesis
gs.init(
    backend=gs.cuda,
    seed=None,
    precision='32',
    debug=False,
    eps=1e-12,
    logging_level=None,
    theme='dark',
    logger_verbose_time=False
)

# create a scene
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5), 
        camera_lookat=(0.0, 0.0, 0.5), 
        camera_fov=30, 
        max_FPS=600
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

# camera setup
cam = scene.add_camera(
    pos=(3, -1, 1.5),
    lookat=(0.0, 0.0, 0.1),
    res=(1280, 960),
    fov=30,
    GUI=False
)

# build
scene.build()

render_output = cam.render(rgb=True, depth=True, segmentation=False, normal=True)
print("Render output length:", len(render_output))
print("Render output types:", [type(x) for x in render_output])

# Unpack 4 values
rgb, depth, _, normal_map = render_output  # Ignore 3rd element (None)

# Verify shapes
print("RGB shape:", rgb.shape)         
print("Depth shape:", depth.shape)    
print("Normal map shape:", normal_map.shape)  

# Find the cylinder’s top face (approximate center pixel)
# Assume cylinder top is near image center (adjust based on actual view)
center_x, center_y = 1280 // 2, 960 // 2  # Image center
normal_pixel = normal_map[center_y, center_x]  # RGB value at center

# Convert RGB normal to vector (Genesis normal map is [-1, 1] range)
normal_vector = (normal_pixel * 2) - 1  # Scale from [0, 1] to [-1, 1]
normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize
print("Normal vector at center (camera space):", normal_vector)

# Start recording
# cam.start_recording()

for i in range(200):
    scene.step()
    cam.render()
 
# cam.stop_recording(save_to_filename='get_normal_by_image_video.mp4', fps=60)