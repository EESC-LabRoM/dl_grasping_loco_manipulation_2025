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
        # euler=[0,90,0],   # Just tresting the effectiveness

    )
)

# camera setup
cam = scene.add_camera(
    pos=(3, -1, 1.5),
    lookat=(0.0, 0.0, 0.5),
    res=(1280, 960),
    fov=30,
    GUI=True
)

# build
scene.build()

# Get cylinder properties
cylinder_position = np.array(cylinder_pos) 
cylinder_height = 0.1
cylinder_quat_tensor = cylinder.get_quat()  
cylinder_quat = cylinder_quat_tensor.cpu().numpy()  # Convert to NumPy array on CPU

# Default Z-axis direction (unrotated cylinder axis)
z_axis = np.array([0, 0, 1])

# Convert quaternion to rotation matrix (if rotated)
def quat_to_rot_matrix(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

# Apply rotation to Z-axis to get cylinder axis
rotation_matrix = quat_to_rot_matrix(cylinder_quat)
cylinder_axis = rotation_matrix.dot(z_axis)  # Rotate [0, 0, 1] by quaternion
cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)  # Normalize

# Top face normal is the cylinder axis (upward direction)
top_face_normal = cylinder_axis
print("Calculated top face normal (world space):", top_face_normal)

# Verify with positions
top_center = cylinder_position + cylinder_axis * cylinder_height  # [0.0, 0.0, 0.15]
bottom_center = cylinder_position
calculated_axis = (top_center - bottom_center) / np.linalg.norm(top_center - bottom_center)
print("Top face normal from positions (world space):", calculated_axis)

# Start recording
# cam.start_recording()

for i in range(120):
    scene.step()
    cam.render()

# cam.stop_recording(save_to_filename='cylinder_normal_calc_via_Numpy_video.mp4', fps=60)