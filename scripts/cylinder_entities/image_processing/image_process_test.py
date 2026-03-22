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
    lookat=(0.0, 0.0, 0.5),
    res=(1280, 960),
    fov=30,
    GUI=False  # Try False to ensure full render data
)

# build
scene.build()

# render rgb, depth, and normal map
render_output = cam.render(rgb=True, depth=True, normal=True, segmentation=False)
print("Render output:", render_output)  # Debug full output
print("Length of render output:", len(render_output))

# Unpack based on actual output
if len(render_output) == 3:
    rgb, depth, normal_map = render_output
    print("RGB shape:", rgb.shape)
    print("Depth shape:", depth.shape)
    if normal_map is not None:
        print("Normal map shape:", normal_map.shape)
        # Extract normal at center
        center_x, center_y = 1280 // 2, 960 // 2
        normal_pixel = normal_map[center_y, center_x]
        normal_vector = (normal_pixel * 2) - 1
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        print("Normal vector at center (camera space):", normal_vector)
    else:
        print("Warning: Normal map is None, falling back to geometric method")
else:
    print("Unexpected render output, falling back to geometric method")
    rgb, depth = render_output  # Handle 2-value case if normal fails

# Start recording
cam.start_recording()

for i in range(120):
    scene.step()
    cam.render()

cam.stop_recording(save_to_filename='cylinder_video.mp4', fps=60)