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
        camera_pos=(0, 0, 1.0),
        camera_lookat=(0, 0, 0.15),
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
    pos=(3, -1, 1.0),
    lookat=(0, 0, 0.15),
    res=(1280, 960),
    fov=30,
    GUI=False
)

# build
scene.build()

# render
render_output = cam.render(
    rgb=True, 
    depth=True, 
    segmentation=False, 
    normal=True
     )
rgb, depth, _, normal_map = render_output

# Verify shapes
print("RGB shape:", rgb.shape)
print("Depth shape:", depth.shape)
print("Normal map shape:", normal_map.shape)

# Debug depth at center
center_x, center_y = 1280 // 2, 960 // 2
print("Center depth:", depth[center_y, center_x])

# Find top face pixels (adjust tolerance based on actual depth)
top_pixels = np.where(np.isclose(depth, 0.85, atol=0.1))  # Wider tolerance
if top_pixels[0].size > 0:
    # Average normals across top face pixels
    normal_vectors = []
    for y, x in zip(top_pixels[0][:10], top_pixels[1][:10]):  # Sample first 10 pixels
        normal_pixel = normal_map[y, x]
        normal_vector = (normal_pixel / 255.0) * 2 - 1
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        normal_vectors.append(normal_vector)
        print(f"Pixel (x={x}, y={y}) - Depth: {depth[y, x]:.3f}, Normal RGB: {normal_pixel}, Normal: {normal_vector}")
    
    # Average normal
    avg_normal = np.mean(normal_vectors, axis=0)
    avg_normal = avg_normal / np.linalg.norm(avg_normal)
    print("Average top face normal (camera space):", avg_normal)
else:
    print("No top face pixels found - check depth range:", depth.min(), depth.max())

# Start recording
# cam.start_recording()

for i in range(200):
    scene.step()
    cam.render()

# cam.stop_recording(save_to_filename='check_normal_by_image_video.mp4', fps=60)