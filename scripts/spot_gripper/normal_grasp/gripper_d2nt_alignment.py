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
from trimesh.transformations import quaternion_from_euler
import random

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
        file='/home/nexus/Desktop/Genesis/genesis/assets/urdf/spot_arm/urdf/open_gripper.urdf',
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
def vector_normalization(normal, eps=1e-8):
    mag = np.linalg.norm(normal, axis=2)
    normal /= np.expand_dims(mag, axis=2) + eps
    return normal

def get_filter(Z, cp2tv=False):
    kernel_Gx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    kernel_Gy = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
    cp2tv_Gx = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
    cp2tv_Gy = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
    if cp2tv:
        Gu = cv2.filter2D(Z, -1, cp2tv_Gx)
        Gv = cv2.filter2D(Z, -1, cp2tv_Gy)
    else:
        Gu = cv2.filter2D(Z, -1, kernel_Gx) / 2
        Gv = cv2.filter2D(Z, -1, kernel_Gy) / 2
    return Gu, Gv

def get_DAG_filter(Z, base=np.e, lap_conf="1D-DLF"):
    gradient_l = np.array([[-1, 1, 0]])
    gradient_r = np.array([[0, -1, 1]])
    gradient_u = np.array([[-1], [1], [0]])
    gradient_d = np.array([[0], [-1], [1]])
    grad_l = cv2.filter2D(Z, -1, gradient_l)
    grad_r = cv2.filter2D(Z, -1, gradient_r)
    grad_u = cv2.filter2D(Z, -1, gradient_u)
    grad_d = cv2.filter2D(Z, -1, gradient_d)
    
    if lap_conf == "1D-DLF":
        lap_hor = abs(grad_l - grad_r)
        lap_ver = abs(grad_u - grad_d)
    else:
        raise ValueError("Only 1D-DLF supported for simplicity")
    
    lambda_map1, lambda_map2 = soft_min(lap_hor, base, 0)
    lambda_map3, lambda_map4 = soft_min(lap_ver, base, 1)
    
    eps = 1e-8
    thresh = base
    lambda_map1[lambda_map1 / (lambda_map2 + eps) > thresh] = 1
    lambda_map2[lambda_map1 / (lambda_map2 + eps) > thresh] = 0
    lambda_map1[lambda_map2 / (lambda_map1 + eps) > thresh] = 0
    lambda_map2[lambda_map2 / (lambda_map1 + eps) > thresh] = 1
    lambda_map3[lambda_map3 / (lambda_map4 + eps) > thresh] = 1
    lambda_map4[lambda_map3 / (lambda_map4 + eps) > thresh] = 0
    lambda_map3[lambda_map4 / (lambda_map3 + eps) > thresh] = 0
    lambda_map4[lambda_map4 / (lambda_map3 + eps) > thresh] = 1
    
    Gu = lambda_map1 * grad_l + lambda_map2 * grad_r
    Gv = lambda_map3 * grad_u + lambda_map4 * grad_d
    return Gu, Gv

def soft_min(laplace_map, base, direction):
    h, w = laplace_map.shape
    eps = 1e-8
    lap_power = np.power(base, -laplace_map)
    if direction == 0:  # horizontal
        lap_pow_l = np.hstack([np.zeros((h, 1)), lap_power[:, :-1]])
        lap_pow_r = np.hstack([lap_power[:, 1:], np.zeros((h, 1))])
        return (lap_pow_l + eps * 0.5) / (eps + lap_pow_l + lap_pow_r), (lap_pow_r + eps * 0.5) / (eps + lap_pow_l + lap_pow_r)
    elif direction == 1:  # vertical
        lap_pow_u = np.vstack([np.zeros((1, w)), lap_power[:-1, :]])
        lap_pow_d = np.vstack([lap_power[1:, :], np.zeros((1, w))])
        return (lap_pow_u + eps / 2) / (eps + lap_pow_u + lap_pow_d), (lap_pow_d + eps / 2) / (eps + lap_pow_u + lap_pow_d)

def MRF_optim(depth, n_est, lap_conf="DLF-alpha"):
    h, w = depth.shape
    n_x, n_y, n_z = n_est[:, :, 0], n_est[:, :, 1], n_est[:, :, 2]
    lap_ker_alpha = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    Z_laplace = abs(cv2.filter2D(depth, -1, lap_ker_alpha))
    Z_laplace_stack = np.array([
        np.hstack((np.inf * np.ones((h, 1)), Z_laplace[:, :-1])),
        np.hstack((Z_laplace[:, 1:], np.inf * np.ones((h, 1)))),
        np.vstack((np.inf * np.ones((1, w)), Z_laplace[:-1, :])),
        np.vstack((Z_laplace[1:, :], np.inf * np.ones((1, w)))),
        Z_laplace
    ])
    best_loc_map = np.argmin(Z_laplace_stack, axis=0)
    Nx_t_stack = np.array([
        np.hstack((np.zeros((h, 1)), n_x[:, :-1])),
        np.hstack((n_x[:, 1:], np.zeros((h, 1)))),
        np.vstack((np.zeros((1, w)), n_x[:-1, :])),
        np.vstack((n_x[1:, :], np.zeros((1, w)))),
        n_x
    ]).reshape(5, -1)
    Ny_t_stack = np.array([
        np.hstack((np.zeros((h, 1)), n_y[:, :-1])),
        np.hstack((n_y[:, 1:], np.zeros((h, 1)))),
        np.vstack((np.zeros((1, w)), n_y[:-1, :])),
        np.vstack((n_y[1:, :], np.zeros((1, w)))),
        n_y
    ]).reshape(5, -1)
    Nz_t_stack = np.array([
        np.hstack((np.zeros((h, 1)), n_z[:, :-1])),
        np.hstack((n_z[:, 1:], np.zeros((h, 1)))),
        np.vstack((np.zeros((1, w)), n_z[:-1, :])),
        np.vstack((n_z[1:, :], np.zeros((1, w)))),
        n_z
    ]).reshape(5, -1)
    n_x = Nx_t_stack[best_loc_map.reshape(-1), np.arange(h * w)].reshape(h, w)
    n_y = Ny_t_stack[best_loc_map.reshape(-1), np.arange(h * w)].reshape(h, w)
    n_z = Nz_t_stack[best_loc_map.reshape(-1), np.arange(h * w)].reshape(h, w)
    return cv2.merge((n_x, n_y, n_z))

def compute_d2nt_normal(depth, version="d2nt_v3"):
    h, w = depth.shape
    # Normalize depth to 0-255 range (Genesis depth is in meters, adjust scale)
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    
    # Fake camera intrinsics (adjust based on Genesis camera if known)
    fx, fy, u0, v0 = 1280 / (2 * np.tan(np.deg2rad(30) / 2)), 960 / (2 * np.tan(np.deg2rad(30) / 2)), 640, 480
    u_map = np.ones((h, 1)) * np.arange(1, w + 1) - u0
    v_map = np.arange(1, h + 1).reshape(h, 1) * np.ones((1, w)) - v0
    
    # Compute gradients
    if version == "d2nt_basic":
        Gu, Gv = get_filter(depth_normalized)
    else:
        Gu, Gv = get_DAG_filter(depth_normalized)
    
    # Depth to Normal Translation
    est_nx = Gu * fx
    est_ny = Gv * fy
    est_nz = -(depth_normalized + v_map * Gv + u_map * Gu)
    est_normal = cv2.merge((est_nx, est_ny, est_nz))
    est_normal = vector_normalization(est_normal)
    
    # MRF optimization for v3
    if version == "d2nt_v3":
        est_normal = MRF_optim(depth_normalized, est_normal)
    
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
output_dir = Path("../Exp_RigidEntity/dataset/analysis")
depth = np.load(output_dir / "iso_depth.npy")
cylinder_pixel_coords = np.load(output_dir / "iso_cylinder_pixel_coords.npy")

# Compute Normal Map using D2NT ~ it is aligning but with some disturbances
normal_map_d2nt = compute_d2nt_normal(depth, version="d2nt_v3")
print(f"D2NT Map: \033[1;92m{normal_map_d2nt}\033[0m")

img_width, img_height = 1280, 960
# center image point
# center_x, center_y = img_width // 2,  img_height // 2

# Select a random cylinder pixel
random_idx = random.randint(0, len(cylinder_pixel_coords) - 1)
random_y, random_x = cylinder_pixel_coords[random_idx]
specific_point_depth = depth[random_y, random_x]
specific_point_normal = normal_map_d2nt[random_y, random_x]  # Use D2NT normal (already [-1, 1])

quartenion = quaternion_from_euler(* vector_to_euler(specific_point_normal))

print(f"Random pixel coordinate: \033[1;92m({random_x}, {random_y})\033[0m")

cam_pos = [1, 1, 1]  
cam_lookat = [0.0, 0.0, 0.09]
fov = 30
up = [0, 0, 1]

grasp_point = pixel_to_world(random_x, random_y, specific_point_depth, cam_pos, cam_lookat, fov, img_width, img_height, up)
print(f"World coordinates: \033[1;92m{grasp_point}\033[0m")

input("\033[1;92mPress Enter to continue. . .\033[0m")

print(f"Orientation: \033[1;92m{quartenion}\033[0m")
print(f"Normal direction: \033[1;92m{specific_point_normal}\033[0m")

for _ in range(200):
    scene.step()

spot_gripper.set_quat(quartenion)
spot_gripper.set_pos(grasp_point + 0.3 * specific_point_normal)

print(f"Gripper Position: {spot_gripper.get_pos()}")

for _ in range(200):
    scene.step()  # Offset along normal

input("\033[1;92mPress Enter to continue. . .\033[0m")

