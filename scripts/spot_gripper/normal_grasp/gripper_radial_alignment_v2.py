import numpy as np
import genesis as gs
import quaternion   # Add this import
from trimesh.transformations import quaternion_matrix, quaternion_from_euler

from genesis.ext.trimesh.transformations import euler_from_matrix, euler_matrix
from scipy.spatial.transform import Rotation as R

# init genesis
gs.init(
    backend             = gs.cuda,
    seed                = None,
    precision           = '32',
    debug               = False,
    eps                 = 1e-16,
    logging_level       = None,
    theme               = 'dark',
    logger_verbose_time = False
    )

# create a scene 
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30, max_FPS=600),
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
        # euler=[0, 90, 0]
    )
)

spot_gripper = scene.add_entity(
    gs.morphs.URDF(
        file='/home/nexus/Desktop/Genesis/genesis/assets/urdf/spot_arm/urdf/open_gripper.urdf',
        euler=(90, 0, 0),
        pos=(-0.2, 0.0, 0.10),
        scale=1.0,
        merge_fixed_links=True,
        fixed = True
    ),
)
# camera setup
# cam = scene.add_camera(
#     pos    = (3, -1, 1.5),
#     lookat = (0.0, 0.0, 0.5),
#     res    = (1280, 960),
#     fov    = 30,
#     GUI    = False
# )
# Build scene
scene.build()

# Set PD control parameters
spot_gripper.set_dofs_kp(
    np.array([100]*2)
    )
spot_gripper.set_dofs_kv(
    np.array([1]*2)
    )
spot_gripper.set_dofs_force_range(
    np.array([-100]*2),
    np.array([100]*2)
    )

print("Aligning...")
def reset_cylinder():
    for i in range(150):
        scene.step()
        # if i % 5: 
        #     cam.render()
    cylinder.set_pos(np.array(cylinder_pos), zero_velocity=True)
    cylinder.set_quat(cylinder_quat, zero_velocity=True)

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])
# Function to convert vector to Euler angles
def vector_to_euler(vec):
    vec = vec / np.linalg.norm(vec)
    reference = np.array([1, 0, 0])
    if np.allclose(vec, reference):
        return np.zeros(3)
    axis = np.cross(reference, vec)
    axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-6 else np.array([0, 0, 1])
    angle = np.arccos(np.dot(reference, vec))
    return R.from_rotvec(angle * axis).as_euler('xyz')

# Get cylinder mesh and find a side face
mesh = cylinder.vgeoms[0].get_trimesh()
side_face_indices = [i for i, n in enumerate(mesh.face_normals) if abs(n[2]) < 1e-6]
if not side_face_indices:
    raise ValueError("No side faces found in the cylinder mesh!")
face_index = side_face_indices[0]

# Compute face centroid and normal in world coordinates
face_local_normal = mesh.face_normals[face_index]
face_local_centroid = np.mean(mesh.vertices[mesh.faces[face_index]], axis=0)
cylinder_quat = cylinder.get_quat().cpu().numpy()
orientation_matrix = quaternion_matrix(cylinder_quat)[:3, :3]
face_world_normal = orientation_matrix @ face_local_normal
face_world_centroid = orientation_matrix @ face_local_centroid + cylinder_pos

# Set gripper position with Z constraint
gripper_pos = face_world_centroid + 0.2 * face_world_normal
gripper_pos[2] = max(gripper_pos[2], 0.01)  # Ensure Z >= 0.01

# Set gripper orientation with additional rotation for side grasp
q_align = quaternion_from_euler(*vector_to_euler(-face_world_normal))
q_rot = quaternion_from_euler(np.pi/2, 0, 0)  # 90-degree rotation around local Z
# gripper_quat = np.quaternion(*q_align) * np.quaternion(*q_rot)
# gripper_quat_array = np.array([gripper_quat.w, gripper_quat.x, gripper_quat.y, gripper_quat.z])
gripper_quat = quaternion_multiply(q_align, q_rot)

# Debugging output
print(f"Cylinder quaternion: {cylinder_quat}")
print(f"Face local normal: {face_local_normal}")
print(f"Face world normal: {face_world_normal}")
print(f"Face world centroid: {face_world_centroid}")
print(f"Gripper position: {gripper_pos}")
print(f"Gripper quaternion: {gripper_quat}")
print(f"q_align Datatype: {q_align.dtype}")
print(f"q_rot Datatype: {q_rot.dtype}")

reset_cylinder()
# Apply gripper pose
spot_gripper.set_pos(gripper_pos)
spot_gripper.set_quat(gripper_quat)
# spot_gripper.set_quat(gripper_quat_array)

# Run simulation
for _ in range(400):
    scene.step()

input("Press Enter to continue...")