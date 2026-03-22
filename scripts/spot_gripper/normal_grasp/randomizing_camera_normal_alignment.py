import numpy as np
import genesis as gs
import random
import quaternion
from trimesh.transformations import quaternion_matrix, quaternion_from_euler
import cv2
from genesis.ext.trimesh.transformations import euler_from_matrix, euler_matrix
from scipy.spatial.transform import Rotation as R

from genesis.options.vis import VisOptions

# init 
gs.init(backend=gs.cuda, precision='32', theme='dark', eps=1e-12)

# create a scene 
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30, max_FPS=600),
    sim_options=gs.options.SimOptions(dt=0.001),
    vis_options=VisOptions(show_world_frame=False, show_link_frame=False,segmentation_level='link'),

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
cam = scene.add_camera(
    pos=(0, 0, 1),
    lookat = (0.0, 0.0, 0.15),
    res    = (1280, 960),
    fov    = 30,
    GUI    = False,
)

# build 
scene.build()
#cam.start_recording()

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
# Grasp parameters
finger_length = 0.09
num_steps = 5
pause_steps = 10

print("\nGetting the normal with image data...")

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

def invert_euler(euler_angles):
    return (euler_angles + np.pi) % (2 * np.pi) - np.pi

def perform_grasp():
    print("\033[1;92mGrasping...\033[0m")
    current_qpos = spot_gripper.get_dofs_position()
    start_gripper_pos = current_qpos[finger_dof].item()  # Should be 0 (open)
    end_gripper_pos = (cylinder_radius / finger_length) * np.pi # Closed position
    gripper_step_size = (end_gripper_pos - start_gripper_pos) / num_steps

    print("\033[1;92mExecuting paused grasp...\033[0m")
    for i in range(num_steps + 1):
        current_gripper_pos = start_gripper_pos + i * gripper_step_size
        target_qpos = current_qpos.clone()
        target_qpos[finger_dof] = current_gripper_pos
        spot_gripper.control_dofs_position(target_qpos)
        print(f"Step {i}: Gripper at {current_gripper_pos:.3f}")
        for _ in range(pause_steps):
            scene.step()
            # if i % 5: 
            #     cam.render()
    print("Stabilizing grasp...")
    for i in range(150):
        scene.step()

def check_gripper_contacts():
    contacts = spot_gripper.get_contacts()
    print(f"\nRaw contacts data: {contacts}")
    
    if not contacts or 'position' not in contacts:
        print("No contacts detected or invalid contact data :/")
        return False
    
    # Use length of 'position' array for number of contacts
    num_contacts = len(contacts['position'])
    print(f"\033[1;92mNumber of contacts detected: {num_contacts}\033[0m")
    
    # Get link names for mapping
    link_names = [l.name for l in scene.rigid_solver.links]
    print(f"Link names in scene: {link_names}")
    
    cylinder_link_idx = link_names.index("cylinder_baselink") if "cylinder_baselink" in link_names else -1
    
    # Process each contact
    cylinder_contact = False
    for i in range(num_contacts):
        print(f"Contact {i + 1}:")
        print(f"  Position: {contacts['position'][i]}")
        # Inferring force_b normal by force direction 
        normal = contacts['force_b'][i] / np.linalg.norm(contacts['force_b'][i])
        print(f"  Normal (inferred): \033[1;92m{normal}\033[0m")
        print(f"  Force on gripper: \033[1;92m{contacts['force_a'][i]}\033[0m")
        print(f"  Force on other: \033[1;92m{contacts['force_b'][i]}\033[0m")
        link_a_idx = contacts['link_a'][i]
        link_b_idx = contacts['link_b'][i]
        print(f"  Link A (gripper): \033[1;92m{link_names[link_a_idx] if link_a_idx < len(link_names) else 'Unknown'}\033[0m")
        print(f"  Link B (other): \033[1;92m{link_names[link_b_idx] if link_b_idx < len(link_names) else 'Unknown'}\033[0m")
        
        if link_b_idx == cylinder_link_idx:
            print("\033[1;92m  -> Contact with cylinder detected!\033[0m")
            cylinder_contact = True
    
    return cylinder_contact

def pixel_to_world(x, y, depth, cam_pos, cam_lookat, fov, img_width, img_height):
    """
    Convert pixel coordinates and depth to world coordinates
    
    Args:
        x: pixel x-coordinate (horizontal)
        y: pixel y-coordinate (vertical)
        depth: depth value at that pixel
        cam_pos: camera position (x, y, z)
        cam_lookat: point camera is looking at (x, y, z)
        fov: field of view in degrees
        img_width: width of the image in pixels
        img_height: height of the image in pixels
    
    Returns:
        world_coords: (x, y, z) in world coordinates
    """
    # Convert inputs to numpy arrays
    cam_pos = np.array(cam_pos)
    cam_lookat = np.array(cam_lookat)
    
    # Calculate camera direction vector
    cam_dir = cam_lookat - cam_pos
    cam_dir = cam_dir / np.linalg.norm(cam_dir)
    
    # Calculate camera up vector (assuming world up is (0,0,1))
    world_up = np.array([0, 0, 1])
    cam_right = np.cross(cam_dir, world_up)
    cam_right = cam_right / np.linalg.norm(cam_right)
    cam_up = np.cross(cam_right, cam_dir)
    
    # Convert FOV to radians
    fov_rad = np.deg2rad(fov)
    
    # Calculate pixel coordinates in normalized device coordinates (-1 to 1)
    ndc_x = (2.0 * x / (img_width - 1)) - 1.0
    ndc_y = 1.0 - (2.0 * y / (img_height - 1))  # Flip y-axis
    
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
def is_in_xy_plane(quaternion):

    current_quat_array = quaternion.cpu().numpy()
    world_z_axis = np.array([0, 0, 1])
    world_z_quat = quaternion_from_euler(*vector_to_euler(-world_z_axis))
    validation = np.allclose(current_quat_array, world_z_quat, atol=1e-2)
    print(f"Comparing the Gripper with Z: \033[1;92m{validation}\033[0m")
    if validation == False:
        print("\033[1;92mGripper is in XY plane, rotating by pi/2...\033[0m")
        q_rot = quaternion_from_euler(np.pi/2, 0, 0)  
        print(f"Rotation quaternion: \033[1;92m{q_rot}\033[0m")
        gripper_quat = np.quaternion(*current_quat_array) * np.quaternion(*q_rot)
        gripper_quat_array = np.array([gripper_quat.w, gripper_quat.x, gripper_quat.y, gripper_quat.z])
        print(f"New quaternion: \033[1;92m{gripper_quat_array}\033[0m")
        spot_gripper.set_quat(gripper_quat_array)
        spot_gripper.set_pos(0.15 * vector_normal+value)  
        return True     
    else:
        print("\033[1;92mGripper invertical, no need rotation.\033[0m")
        return False

def randomize_camera_position(camera):
    x = random.uniform(-0.5, 0.5)
    y = random.uniform(-0.5, 0.5)
    z = random.uniform(0.16, 0.6)
    camera.set_pose(
        pos = (x, y, z),
        lookat = (0.0, 0.0, 0.15),
        )
    return (x, y, z)

# cam_pos = randomize_camera_position(cam)
# print(f"Camera moved to: ({cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f})")

for i in range(5):
    scene.step()
    cam.set_pose(
        pos = (3, -1.5, 0.2),
        lookat = (0.0, 0.0, 0.09),
    )
    render_output = cam.render(rgb=True, depth=True, segmentation=True, normal=True, colorize_seg=False)


rgb, depth, seg, normal_map = render_output

center_x, center_y = 1280 // 2, 960 // 2
depth_center = depth[center_y, center_x]
print(f"Depth at center: \033[1;92m{depth_center}\033[0m")
normal_pixel = None
specific_point_normal = normal_map[center_y, center_x]
specific_point_depth = depth[center_y, center_x]
# print(f"Normal Map shape: \033[1;92m{normal_map.shape}\033[0m")
# print(scene.rigid_solver.links)

vector_normal = 2*(specific_point_normal - 127.5)/255
value = pixel_to_world(center_x, center_y, specific_point_depth, (3, -1.5, 0.2), (0.0, 0.0, 0.09),  30, img_width=1280, img_height=960)

#print(f"Segmentation links: \033[1;92m{idx}\033[0m")
print(f"Normal vector in Pixels value: \033[1;92m{specific_point_normal}\033[0m")
print(f"Normal vector in Euler: \033[1;92m{vector_to_euler(vector_normal)}\033[0m")
print(f"Point position {value}")
input("\nEnter. . . \n")
for _ in range(200):
    scene.step()
    render_output = cam.render(rgb=True, depth=True, segmentation=True, normal=True)
# TODO: 
# 1. align the spot with 
print("Aligning the gripper with Normal Surface Vector")
# Pixel's quaternion
quartenion = quaternion_from_euler(* vector_to_euler(-vector_normal))
spot_gripper.set_quat(quartenion)
spot_gripper.set_pos(0.3 * vector_normal+value)  
current_pos = spot_gripper.get_pos()
print(f"Gripper current position: \033[1;92m{current_pos}\033[0m")

for _ in range(200):
    scene.step()
    cam.render(rgb=True, depth=True, segmentation=True, normal=True)
# Adjusting gripper orientation for grasping
current_quat = spot_gripper.get_quat()
is_in_xy_plane(current_quat)
for _ in range(200):
    scene.step()
    cam.render()
input("\nEnter. . . \n")
# Grasping
perform_grasp()

for _ in range(200):
    scene.step()
    cam.render()

#Checking for contacts
print(f"\nChecking contacts after grasp... {check_gripper_contacts()}")

for _ in range(200):
    scene.step()
    cam.render()

# cam.stop_recording(save_to_filename="Gripper_Static_Axial_Grasping_video.mp4")