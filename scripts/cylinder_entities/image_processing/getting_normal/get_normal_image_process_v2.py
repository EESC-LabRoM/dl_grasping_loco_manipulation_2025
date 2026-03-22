import genesis as gs
import numpy as np

gs.init(backend=gs.cuda, precision='32', theme='dark', eps=1e-12)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0, 0, 1),
        camera_lookat=(0, 0, 0.15),
        camera_fov=2,
        max_FPS=600
    ),
    sim_options=gs.options.SimOptions(dt=0.001),
    show_viewer=True,
    show_FPS=False
)

plane = scene.add_entity(gs.morphs.Plane())
cylinder = scene.add_entity(
    gs.morphs.Cylinder(pos=[0, 0, 0.05], height=0.1, radius=0.05)
)

camera_pos = np.array([0, 0, 1])
lookat_cam = np.array([0, 0, 0.15])
up = np.array([0, 1, 0])

cam = scene.add_camera(pos=camera_pos, lookat=lookat_cam, up=up, res=(1280, 960), fov=2, GUI=False)
scene.build()

render_output = cam.render(rgb=True, depth=True, segmentation=False, normal=True)
rgb, depth, _, normal_map = render_output

print("Render output length:", len(render_output))
print("Render output types:", [type(x) for x in render_output])
print("RGB shape:", rgb.shape)
print("Depth shape:", depth.shape)
print("Normal map shape:", normal_map.shape)
print("Depth range:", depth.min(), "to", depth.max())

center_x, center_y = 1280 // 2, 960 // 2
depth_center = depth[center_y, center_x]
print("Depth at center:", depth_center)

normal_pixel = None
if abs(depth_center - 0.85) > 0.005:
    print("Center pixel missed top face, searching...")
    for y in range(center_y - 100, center_y + 100):
        for x in range(center_x - 100, center_x + 100):
            if abs(depth[y, x] - 0.85) < 0.005:
                normal_pixel = normal_map[y, x]
                print(f"Found top face at ({x}, {y}), depth: {depth[y, x]}")
                break
        else:
            continue
        break
    if normal_pixel is None:
        print("Top face not found! Using center pixel as fallback.")
        normal_pixel = normal_map[center_y, center_x]
else:
    normal_pixel = normal_map[center_y, center_x]
    print("Center pixel hits top face.")

normal_camera = (normal_pixel * 2) - 1
normal_camera = normal_camera / np.linalg.norm(normal_camera)

# Use camera's transform property
R_world_to_cam = cam.transform  # 4x4 world-to-camera matrix
R_cam_to_world = np.linalg.inv(R_world_to_cam)  # 4x4 camera-to-world
R_cam_to_world_3x3 = R_cam_to_world[:3, :3]  # Extract 3x3 rotation part
normal_world = R_cam_to_world_3x3 @ normal_camera

# Flip Z if necessary to match world up (assuming Genesis normal map Z is toward camera)
normal_world = np.array([normal_world[0], normal_world[1], -normal_world[2]])

print("Camera transform (world-to-camera):\n", R_world_to_cam)
print("Camera-to-world rotation (3x3):\n", R_cam_to_world_3x3)
print("Normal vector at selected pixel (camera space):", normal_camera)
print("Normal in world space:", normal_world)

for _ in range(200):
    scene.step()
    cam.render()