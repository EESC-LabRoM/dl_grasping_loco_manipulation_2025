"""
In this script I tryed to apply the Normal Map Method from the study:
https://github.com/hongfo/depth_to_normal/tree/main  because this will be used
in the deploy test with real Spot Arm.

It's needs to be checked because the normal_vector obtained isn't the
cylinder's radial direction.I suspect that is something in the calculations, 
in the units adopted by the  functions or any reference direction that don't 
match with the Simulated Environment.
"""
import genesis as gs
import numpy as np
from pathlib import Path
import cv2
from scipy.spatial.transform import Rotation as R
from trimesh.transformations import quaternion_from_euler, euler_from_quaternion
import random

from utils import *

# Genesis Setup
gs.init(backend=gs.cuda, precision='32', theme='dark', eps=1e-12)
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(camera_pos=(1, 1, 1), camera_lookat=(0, 0, 0.15), camera_fov=30, max_FPS=600),
    sim_options=gs.options.SimOptions(dt=0.001),
    show_viewer=True,
    show_FPS=False
)

plane = scene.add_entity(gs.morphs.Plane())
cylinder = scene.add_entity(
    gs.morphs.Cylinder(
        pos=[0.0, 0.0, 0.05],
        height=0.1, 
        radius=0.03,
        collision=True
    )
)
spot_gripper = scene.add_entity(
    gs.morphs.URDF(
        file='urdf/spot_arm/urdf/open_gripper.urdf',
        euler=(90, 0, 0),
        pos=(-1, 0.0, 0.10),
        scale=1.0,
        merge_fixed_links=True,
        fixed=True
    )
)
# build 
scene.build()

# D2NT Functions
def compute_d2nt_normal(depth_path, normal_path, VERSION="d2nt_v3"):
    
    # Fake camera intrinsics (adjust based on Genesis camera if known)
    cam_fx, cam_fy, u0, v0 = 1280 / (2 * np.tan(np.deg2rad(30) / 2)), 960 / (2 * np.tan(np.deg2rad(30) / 2)), 640, 480

    # get ground truth normal [-1,1]
    normal_gt = get_normal_gt(normal_path)
    normal_gt = vector_normalization(normal_gt)
    h, w, _ = normal_gt.shape

    # get depth
    depth, mask = get_depth(depth_path, h, w) 
    u_map = np.ones((h, 1)) * np.arange(1, w + 1) - u0
    v_map = np.arange(1, h + 1).reshape(h, 1) * np.ones((1, w)) - v0

    # get depth gradients
    if VERSION == 'd2nt_basic':
        Gu, Gv = get_filter(depth)
    else:
        Gu, Gv = get_DAG_filter(depth)

    # Depth to Normal Translation
    est_nx = Gu * cam_fx
    est_ny = Gv * cam_fy
    est_nz = -(depth + v_map * Gv + u_map * Gu)
    est_normal = cv2.merge((est_nx, est_ny, est_nz))

    # vector normalization
    est_normal = vector_normalization(est_normal)

    # MRF-based Normal Refinement
    if VERSION == 'd2nt_v3':
        est_normal = MRF_optim(depth, est_normal)

    np.save("simulations/d2nt/data/d2nt.npy", est_normal)

    return est_normal

# Convert to world coordinates
def pixel_to_world(pixel_x, pixel_y, depth, cam_pos, cam_lookat, fov, img_width, img_height, world_up):
    # fov_rad = np.deg2rad(fov)
    # fx = fy = img_width / (2 * np.tan(fov_rad / 2))
    # x = (pixel_x - img_width / 2) * depth / fx
    # y = (pixel_y - img_height / 2) * depth / fy
    # z = depth
    # world_x = cam_pos[0] + x
    # world_y = cam_pos[1] + y
    # world_z = cam_pos[2] - z
    # return np.array([world_x, world_y, world_z])
    cam_pos = np.array(cam_pos)
    cam_lookat = np.array(cam_lookat)
    
    # Calculate camera direction vector
    cam_dir = cam_lookat - cam_pos
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    
    # Calculate camera up vector (assuming world up is (0,0,1))
    world_up = np.array(world_up)
    cam_right = np.cross(cam_dir, world_up)
    cam_right = cam_right / np.linalg.norm(cam_right)
    cam_up = np.cross(cam_right, cam_dir)
    
    # Convert FOV to radians
    fov_rad = np.deg2rad(fov)
    
    # Calculate pixel coordinates in normalized device coordinates (-1 to 1)
    ndc_x = (2.0 * pixel_x / (img_width - 1)) - 1.0
    ndc_y = 1.0 - (2.0 * pixel_y / (img_height - 1))  # Flip y-axis
    
    # Calculate direction vector in camera space
    aspect_ratio = img_width / img_height
    tan_half_fov = np.tan(fov_rad / 2.0)
    
    cam_space_x = ndc_x * tan_half_fov * aspect_ratio
    cam_space_y = ndc_y * tan_half_fov
    cam_space_z = -1.0  # Negative z is forward in camera space
    
    # Create direction vector in camera space
    cam_space_dir = np.array([cam_space_x, cam_space_y, cam_space_z])
    cam_space_dir = cam_space_dir / np.linalg.norm(cam_space_dir)
    
    # Convert to world space direction
    world_dir = (cam_right * cam_space_dir[0] + 
                cam_up * cam_space_dir[1] + 
                -cam_dir * cam_space_dir[2])
    world_dir = world_dir / np.linalg.norm(world_dir)
    
    # Calculate world position
    world_pos = cam_pos + world_dir * depth
    
    return world_pos

def normalize(v):
    return v / np.linalg.norm(v)

def rotate_vector(normal, cam_pos, cam_lookat, up):
    # Vetor direção da câmera (câmera olha de cam_pos para cam_lookat)
    z_cam = normalize(np.array(cam_pos) - np.array(cam_lookat))
    
    # Vetor "para cima" definido pelo usuário
    up = np.array(up)
    
    # Vetor X da câmera, perpendicular ao Z (produto vetorial de up e z_cam)
    x_cam = normalize(np.cross(up, z_cam))
    
    # Vetor Y da câmera, perpendicular ao plano XZ (produto vetorial de z_cam e x_cam)
    y_cam = np.cross(z_cam, x_cam)

    rotation_matrix = np.column_stack([x_cam, y_cam, z_cam])

    # Aplicando a transformação nos vetores normais (multiplicação matricial)
    transformed_normals = np.dot(normal, rotation_matrix)  # Transposta de R para a transformação correta

    return transformed_normals

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
    print(rotation)
    euler_angles = rotation.as_euler('xyz')
    
    return euler_angles

def rot_z(vec, theta):
    # Converter o ângulo para radianos
    angulo_rad = np.radians(theta)
    
    # Matriz de rotação em torno de Z
    matriz_rotacao = np.array([
        [np.cos(angulo_rad), -np.sin(angulo_rad), 0],
        [np.sin(angulo_rad), np.cos(angulo_rad), 0],
        [0, 0, 1]
    ])
    
    # Rotacionando o vetor
    vetor_rotacionado = np.dot(matriz_rotacao, vec)
    
    return vetor_rotacionado

# Define DOFs
hand_dof = np.arange(2)
finger_dof = np.array([1]) 

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

# Load Genesis Data
data = 'simulations/d2nt/data/'
depth_path = data+'side_depth.npy'
normal_path = data+'side_rgb_cylinder.png'

cylinder_pixels = data+'side_cylinder_pixel_coords.npy'
cylinder_pixel_coords = np.load(cylinder_pixels)

depth = np.load(depth_path)

# Compute Normal Map using D2NT
normal_map_d2nt = compute_d2nt_normal(depth_path,normal_path)
np.save('d2nt_normal_data.npy',normal_map_d2nt)

# Select a random cylinder pixel
random_idx = random.randint(0, len(cylinder_pixel_coords) - 1)
random_y, random_x = cylinder_pixel_coords[random_idx]
specific_point_depth = depth[random_y, random_x]
specific_point_normal = normal_map_d2nt[random_y, random_x]  # Use D2NT normal (already [-1, 1])
print(f"Random pixel coordinate: \033[1;92m({random_x}, {random_y})\033[0m")


# Camera
cam_pos = [1, 0, 0.05]  
cam_lookat = [0.0, 0.0, 0.05]
fov = 30
img_width, img_height = 1280, 960
up = [0,0,1]

specific_point_normal = rotate_vector(specific_point_normal, cam_pos, cam_lookat, up)
quartenion = quaternion_from_euler(* vector_to_euler(-1*specific_point_normal))

grasp_point = pixel_to_world(random_x, random_y, specific_point_depth, cam_pos, cam_lookat, fov, img_width, img_height, up)
print(f"World coordinates: \033[1;92m{grasp_point}\033[0m")

position = grasp_point + 0.3 * (specific_point_normal+grasp_point)

input("\033[1;92mPress Enter to continue. . .\033[0m")

print(f"\033[1;92m{"_______________________________________________________"}\033[0m")

print(f"Orientation: \033[1;92m{euler_from_quaternion(quartenion)}\033[0m")

print(f"\033[1;92m{"_______________________________________________________"}\033[0m")

print(f"Grasp Point: \033[1;92m{grasp_point}\033[0m")

print(f"Normal specific Point: \033[1;92m{specific_point_normal}\033[0m")

print(f"Gripper Graspping Point: \033[1;92m{position}\033[0m")

print(f"\033[1;92m{"_______________________________________________________"}\033[0m")

for _ in range(200):
    scene.step()

spot_gripper.set_quat(quartenion)
spot_gripper.set_pos(position)

print(f"Gripper Position: {spot_gripper.get_pos()}")

for _ in range(200):
    scene.step()  # Offset along normal

input("\033[1;92mPress Enter to continue. . .\033[0m")

