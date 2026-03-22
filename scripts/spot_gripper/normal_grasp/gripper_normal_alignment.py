import numpy as np
import genesis as gs
from trimesh.transformations import quaternion_matrix, quaternion_from_euler

from genesis.ext.trimesh.transformations import euler_from_matrix, euler_matrix
from scipy.spatial.transform import Rotation as R

def vector_to_euler(vec):
    vec = np.array(vec) / np.linalg.norm(vec)  # Normalize the vector
    reference = np.array([1, 0, 0])  # Reference vector
    
    if np.allclose(vec, reference):
        return np.array([0.0, 0.0, 0.0])
    
    axis = np.cross(reference, vec)
    angle = np.arccos(np.dot(reference, vec))
    
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([0, 0, 1])
    else:
        axis = axis / np.linalg.norm(axis)
    
    rotation = R.from_rotvec(angle * axis)
    euler_angles = rotation.as_euler('xyz')
    
    return euler_angles

# init 
gs.init(backend=gs.cuda, precision='32', theme='dark', eps=1e-12)

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

def normal_to_euler(normal):
    # Extract components
    nx, ny, nz = normal
    # Compute Euler angles (XYZ order)
    # Pitch (θ): angle from Z-axis
    pitch = np.arccos(nz)  # Radians
    # Yaw (ψ): angle in XY plane
    yaw = np.arctan2(ny, nx)  # Radians
    # Roll (φ): Set to 0 (no constraint from normal alone)
    roll = 0.0
    return [roll, pitch, yaw]


# build 
scene.build()
# cam.start_recording()

print(spot_gripper)
print(spot_gripper.n_dofs)

# Define DOFs
hand_dof = np.arange(2)
finger_dof = np.array([1]) 

# Set PD control parameters
spot_gripper.set_dofs_kp(
    np.array([100, 100])
    )
spot_gripper.set_dofs_kv(
    np.array([ 1, 1])
    )
spot_gripper.set_dofs_force_range(
    np.array([ -100, -100]),
    np.array([ 100, 100])
    )

print("Grasping...")

def invert_euler(euler_angles):
    return (euler_angles + np.pi) % (2 * np.pi) - np.pi

# Specify the cylinder face: 

#list of visual geometry objects for the cylinder|[0] is the first (likely only) mesh|turns it into a Trimesh object
mesh = cylinder.vgeoms[0].get_trimesh() 
# Picks the last face in the mesh (top face)
face_index = -1 
# Gets the normal vector of that face in local space
face_local_normal = mesh.face_normals[face_index] 
# Gets the indices of the 3 vertices making up this triangular face.
vertex_indices = mesh.faces[face_index]
# Gets the 3D positions of those vertices in local space 
vertex_positions = mesh.vertices[vertex_indices] 
# Averages the vertices to find the face’s center
face_local_centroid = np.mean(vertex_positions, axis=0) # 

# Transform the local face centroid to world coordinates:

# Converts the cylinder’s quaternion into a 4x4 rotation matrix and takes only the 3x3x
cylinder_orientation_matrix = quaternion_matrix(cylinder.get_quat().cpu().numpy())[:3, :3]
# Twists the local centroid using matrix
rotated_centroid = np.dot(cylinder_orientation_matrix, face_local_centroid)
# Adds the cylinder’s world position
face_world_centroid = rotated_centroid + cylinder.get_pos().cpu().numpy()
# Transform the local face normal to world coordinates
face_world_normal = np.dot(cylinder_orientation_matrix, face_local_normal)
# Get the quartenion of orientation 
quartenion = quaternion_from_euler(* vector_to_euler(-face_world_normal))

print("Cylinder quart: ", cylinder.get_quat().cpu().numpy())
print("Face local normal: ", face_local_normal)
print("Face world normal: ", face_world_normal)
print("Face world center: ", face_world_centroid)
print("Face world quart: ", quartenion)

# Aligning the gripper with the direction obtained
spot_gripper.set_quat(quartenion)
spot_gripper.set_pos(face_world_centroid + 0.3 * face_world_normal)
current_qpos = spot_gripper.get_dofs_position()

# Stabilize
print("Stabilizing...")
for i in range(150):
    scene.step()
    # if i%10:
    #     cam.render()

input("Enter")
# cam.stop_recording(save_to_filename="Gripper_Static_Axial_Grasping_video.mp4")