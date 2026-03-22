import argparse
import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
import torch
from torch import nn 
from torchvision.models import mobilenet_v2
from ultralytics import YOLO
import bosdyn.client
from bosdyn.client import create_standard_sdk
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import (
    RobotCommandClient,
    RobotCommandBuilder,
    blocking_stand,
    block_until_arm_arrives,
)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.util import setup_logging, authenticate, add_base_arguments
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME, WR1_FRAME_NAME
from bosdyn.api import geometry_pb2, arm_command_pb2
from bosdyn.client import math_helpers
from trimesh.transformations import quaternion_from_euler
from scipy.spatial.transform import Rotation as R
from utils import * 

# checks for GPU access with PyTorch
print("PyTorch version: ",torch.__version__)
print("CUDA Available:", torch.cuda.is_available()) 
print("PyTorch-Cuda version: ",torch.version.cuda) 
print("Device name: ", torch.cuda.get_device_name(0))
device = "cuda" if torch.cuda.is_available() else "cpu"

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.cm as cm

def plot_depth_image_with_colorbar(depth_image, name, window_title="Gripper Depth", num_bins=12, max_depth = 1000):
    min_depth = 0
    # Clip values to the defined range for better visualization
    clipped_depth = np.clip(depth_image, min_depth, max_depth)
    bin_edges = np.linspace(min_depth, max_depth, 13)
    binned_depth_indices = np.digitize(clipped_depth, bin_edges, right=False)
    binned_depth_indices = np.clip(binned_depth_indices - 1, 0, 11)
    base_cmap = cm.get_cmap('viridis', 12) # Get a colormap discretized to num_bins colors
    norm = Normalize(vmin=0, vmax=11)
    # Normalize to 0-1 range for colormapping
    # normalized_depth = (clipped_depth - min_depth) / (max_depth - min_depth)
    colormap = plt.cm.viridis
    # depth_display_rgb = colormap(normalized_depth)[..., :3]  # [..., :3] to remove alpha channel

    # --- Matplotlib Plotting ---
    plt.figure(figsize=(20, 16))
    # plt.imshow(binned_depth_indices, cmap=colormap)
    img = plt.imshow(binned_depth_indices, cmap=base_cmap, norm=norm, origin='upper', extent=[0, depth_image.shape[1], depth_image.shape[0], 0])
    # plt.colorbar(label=f"Depth ({'m' if max_depth > 1 else 'mm'})") # Adjust unit based on your data
    cbar = plt.colorbar(img, ticks=np.arange(num_bins) + 0.5, orientation='vertical', label=f"Depth (mm)")
    
    # Set the labels for the colorbar ticks to show the actual depth ranges
    cbar_labels = []
    for i in range(num_bins):
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i+1]
        cbar_labels.append(f"{lower_bound:.2f} - {upper_bound:.2f}")
        
    cbar.ax.set_yticklabels(cbar_labels)
    cbar.ax.tick_params(labelsize=8) # Adjust label size if needed

    plt.title(window_title)
    plt.xlabel("X-axis (pixels)")
    plt.ylabel("Y-axis (pixels)")
    plt.savefig(name)
    


class UpBlock(nn.Module):
    def __init__(self, in_up, in_skip, out_c, groups=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_up, out_c, 2, stride=2, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(out_c + in_skip, out_c, 3, padding=1, groups=groups, bias=False),
            nn.GroupNorm(8, out_c), nn.ReLU(inplace=True)
        )
    def forward(self, x_up, x_skip):
        x = self.up(x_up)
        x = torch.cat([x_skip, x], 1)
        return self.conv(x)

class GraspNN_V3(nn.Module):
    def __init__(self, in_channels=8, base=16):
        super().__init__()
        mb = mobilenet_v2(weights=None)
        # adapta 8 canais para 32
        mb.features[0][0] = nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False)
        self.backbone = mb.features

        self.enc1 = self.backbone[:3]     # 24 ch
        self.enc2 = self.backbone[3:6]    # 32 ch
        self.enc3 = self.backbone[6:10]   # 64 ch   (skip)
        self.enc4 = self.backbone[10:]    # 1280 ch (bottleneck)

        self.up3 = UpBlock(1280, 64,  base*8)  # <-- canais certos
        self.up2 = UpBlock(base*8, 32, base*4)
        self.up1 = UpBlock(base*4, 24, base*2)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(base*2, base, 2, stride=2, bias=False),
            nn.GroupNorm(8, base), nn.ReLU(inplace=True))
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x  = self.enc4(s3)

        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x)
        return self.outc(x)

def approach_vector_to_euler(vec):
    vec = np.array(vec) / np.linalg.norm(vec)
    reference = np.array([1, 0, 0])
    if np.allclose(vec, reference):
        return np.array([0.0, 0.0, 0.0])
    axis = np.cross(reference, vec)
    angle = np.arccos(np.dot(reference, vec))
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([0, 0, 1])
    else:
        axis = axis / np.linalg.norm(axis)
    rotation = R.from_rotvec(angle * axis)
    euler_angles = rotation.as_euler("xyz")
    return euler_angles

def get_rgb_depth_images(image_client, command_client):
    """Capture RGB and Depth images from Spot's gripper camera."""
    gripper_command = RobotCommandBuilder.claw_gripper_open_command()
    command_client.robot_command(gripper_command)
    time.sleep(3.0)
    print(f"\n\033[1;92mGetting RGB and Depth . . .\033[0m\n"
          f"Gripper opened to maximize camera FOV."
          )

    sources = ["hand_color_image", "hand_depth_in_hand_color_frame"]   # hand_depth_in_hand_color_frame
    responses = image_client.get_image_from_sources(sources)
    if len(responses) != 2:
        print(f"Erro: Expected 2 images, got {len(responses)}")
        return None, None, None, None, None, None

    rgb_response = responses[0]
    if rgb_response.source.name != "hand_color_image":
        print(f"Erro: Expected hand_color_image, got {rgb_response.source.name}")
        return None, None, None, None, None, None
    rgb_img = cv2.imdecode(
        np.frombuffer(rgb_response.shot.image.data, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    print(
        f"RGB Source: {rgb_response.source.name}, "
        f"Resolution: {rgb_img.shape[0]}x{rgb_img.shape[1]}, "
        f"Shape: {rgb_img.shape}"
    )

    depth_response = responses[1]
    if depth_response.source.name != "hand_depth_in_hand_color_frame":
        print(f"Erro: Expected hand_depth_in_hand_color_frame, got {depth_response.source.name}")
        return None, None, None, None, None, None
    depth_data = np.frombuffer(depth_response.shot.image.data, dtype=np.uint16)
    depth_img = depth_data.reshape(depth_response.shot.image.rows, depth_response.shot.image.cols)
    print(
        f"Depth Source: {depth_response.source.name}, "
        f"Resolution: {depth_img.shape[0]}x{depth_img.shape[1]}, "
        f"Shape: {depth_img.shape}"
    )

    pinhole = depth_response.source.pinhole.intrinsics
    fx, fy = pinhole.focal_length.x, pinhole.focal_length.y
    u0, v0 = pinhole.principal_point.x, pinhole.principal_point.y

    # timestamp = time.strftime("%Y%m%d%H%M%S")
    # rgb_filename = f"rgb_{timestamp}.png"
    # depth_filename = f"depth_{timestamp}.npy"
    # cv2.imwrite(rgb_filename, rgb_img)
    # np.save(depth_filename, depth_img)
    # print(f"Saved RGB image: {rgb_filename}")
    # print(f"Saved Depth image: {depth_filename}")

    return rgb_img, depth_img, rgb_response.shot, fx, fy, u0, v0

def segment_bottle(model, rgb_img):
    """Segment a bottle using YOLO and extract mask."""
    print(f"\n\033[1;92mSegmenting bottle . . .\033[0m\n")
    results = model(rgb_img)[0]
    mask = np.zeros_like(rgb_img[:, :, 0], dtype=np.uint8)
    
    if results.masks is None or len(results.boxes) == 0:
        print("No objects detected.")
        cv2.imshow("original image", cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        cv2.imshow("mask", mask * 255)  # Scale for visibility
        time.sleep(3)
        return None

    for i, (cls, conf, m) in enumerate(zip(results.boxes.cls, results.boxes.conf, results.masks)):
        class_name = results.names[int(cls)]
        print(class_name)
        if class_name in ["bottle", "vase"]:
            print(f"Detected {class_name} with confidence {conf:.3f}")
            mask = m.data.cpu().numpy().squeeze()  # Get binary mask
            mask = (mask > 0).astype(np.uint8)  # Ensure binary (0 or 1)
            break  # Use first valid mask
    else:
        print("No bottle or vase detected!")
        cv2.imshow("original image", cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        cv2.imshow("mask", mask * 255)
        time.sleep(3)
        return None

    print(f"Mask shape: {mask.shape}, Non-zero pixels: {np.sum(mask)}")
    cv2.imshow("original image", cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    cv2.imshow("mask", mask * 255)  # Scale for visibility
    time.sleep(3)
    return mask

def preprocess_data(rgb, depth, normal, mask, target):
    print(f"\n\033[1;92mPre-processing data. . .\033[0m")
    depth = np.copy(depth)  # Create a copy to avoid read-only issues
    normal = np.copy(normal)

    not_zero = depth > 100
    not_inf = depth < 800
    dist = depth[(mask != 0) & not_zero & not_inf].mean() / 1000
    
    # Translate depth
    kernel = np.ones((2, 2), np.uint8)  # Set kernel size (adjust as needed)
    
    print(f"    1. Mask step")
    mask = cv2.inRange(mask, 1, 1) # Select only the object, removing the gripper or other elements
    mask = cv2.erode(mask, kernel, iterations=3).astype(int)
    print(f"    2. Depth step")    
    # TODO: review
    depth[mask == 0] = 2000 # Place everything that is not the object 2 meters away
    depth = depth/depth.max() # Normalizes depth
    
    print(f"    3. Normal step")
    normal[mask == 0] = np.array([0.49803922, 0.49803922, 1. ])*255 # Removes unwanted normals
            
    #grasp = (cv2.inRange(grasp, 0, 0).astype(bool) * mask).astype(int) + grasp # Preenche os espaços vazios da garrafa
    
    target = cv2.dilate(target, kernel, iterations=2)
    
    return rgb, depth, normal, mask, dist

def nn_grasping_model(loaded_model, rgb_img, depth_img, mask, normal_map):
    """ Here the data will be passed through the loaded_model """
    print(f"\n\033[1;92mPassing data through the loaded model . . .\033[0m")
    view_rgb_t = torch.from_numpy(np.copy(rgb_img)).permute(2,0,1).float()
    view_depth_t = torch.from_numpy(np.copy(depth_img)).unsqueeze(dim=2).permute(2,0,1).float()
    view_normal_t = torch.from_numpy(np.copy(normal_map)).permute(2,0,1).float()
    view_seg_t = torch.from_numpy(np.copy(mask)).unsqueeze(dim=2).permute(2,0,1).float()
    data = torch.cat([
        view_rgb_t,
        view_depth_t,
        view_normal_t,
        view_seg_t,
        ],dim=0).unsqueeze(dim=0).to(device)
    
    with torch.inference_mode():
        output = loaded_model(data).squeeze().cpu()

    # Ensure grasp point is within mask
    masked_output = output * mask
    if masked_output.max() == 0:
        print("No valid grasp point within mask.")
        return None, None
    
    grasp_y, grasp_x = np.unravel_index(np.argmax(masked_output), output.shape)
    print(f"Grasp point: ({grasp_x}, {grasp_y}), Mask value: {mask[grasp_y, grasp_x]}")
    return grasp_y, grasp_x

def d2nt(depth, fx, fy, u0, v0):
    """normal maping version implemented: d2nt_v3"""
    print(f"\n\033[1;92mApply d2nt method. . .\033[0m")
    h, w = depth.shape
    depth = depth.astype(np.float64) / 1000.0
    u_map = np.ones((h, 1)) * np.arange(1, w + 1) - u0
    v_map = np.arange(1, h + 1).reshape(h, 1) * np.ones((1, w)) - v0
    Gu, Gv = get_DAG_filter(depth)
    est_nx = Gu * fx
    est_ny = Gv * fy
    est_nz = -(depth + v_map * Gv + u_map * Gu)
    est_normal = cv2.merge((est_nx, est_ny, est_nz))
    est_normal = vector_normalization(est_normal)
    est_normal = MRF_optim(depth, est_normal)
    return est_normal

def calculate_params(rgb_img, depth_img, grasp_y, grasp_x, normal_map, fx, fy, u0, v0):
    """Calculate grasp parameters using d2nt and grasp pixel. """
    print(f"\n\033[1;92mCalculating grasping parameters. . .\033[0m")
    x_rgb, y_rgb = grasp_x, grasp_y 
    depth_height, depth_width = depth_img.shape
    rgb_height, rgb_width = rgb_img.shape[:2]

    if rgb_width == depth_width and rgb_height == depth_height:
        x = x_rgb
        y = y_rgb
    else:
        y = int(y_rgb * depth_height / rgb_height)
        x = int(x_rgb * depth_width / rgb_width)

    x = min(max(x, 0), depth_width - 1)
    y = min(max(y, 0), depth_height - 1)
    depth_value = depth_img[y, x]
    print(f"Grasp pixel (x, y): ({x_rgb}, {y_rgb}), Depth at ({x}, {y}): {depth_value} mm")
    # if (depth_value > 800) or (depth_value < 100):
    #     raise ValueError("Erro no sensor de distância")
    
    Z = depth_value / 1000.0  # Convert to meters for position calculation
    if Z <= 0:
        print("Invalid depth (0 mm).")
        return None, None, None, None

    X = (x - u0) * Z / fx
    Y = (y - v0) * Z / fy
    distance = np.sqrt(X**2 + Y**2 + Z**2)
    normal_approach_vector = normal_map[y, x]
    position = (X, Y, Z)

    print(f"d2nt parameters: u0={u0}, v0={v0}, fx={fx}, fy={fy}")
    print(f"Position: ({X:.3f}, {Y:.3f}, {Z:.3f}) m, Distance: {distance:.2f} m, Normal: {normal_approach_vector}")

    return (x_rgb, y_rgb), normal_approach_vector, position, distance

def align_gripper(robot, command_client, position, normal_approach_vector, offset, distance, shot):
    """Align gripper to the computed pose."""
    print(f"\n\033[1;92mAligning gripper. . .\033[0m")
    try:
        hand_frame = "hand_color_image_sensor" 
        camera_tform_body = get_a_tform_b(
            shot.transforms_snapshot, BODY_FRAME_NAME, hand_frame
        )
        print(f"TRANSFORMATION INFO: \n{shot.transforms_snapshot}\n")

        if camera_tform_body is None:
            raise ValueError(
                f"Could not find transform from {hand_frame} to {BODY_FRAME_NAME}"
            )

        normal_approach_vector = np.array([0, 0, -1])
        flipped_normal = [-n for n in normal_approach_vector]

        normal = np.array([flipped_normal[0], flipped_normal[1], flipped_normal[2]])
        # Add a rotation offset (~15 deg)
        n_euler = approach_vector_to_euler(normal)
        euler_offset = np.array([np.deg2rad(-15), np.deg2rad(-15), 0])
        offseted_normal = n_euler# + euler_offset
        # print(f"offsetd normal: \033[1;92m{offseted_normal}\033[0m")
        quat = quaternion_from_euler(*offseted_normal)
        rotation = geometry_pb2.Quaternion(w=quat[0], x=quat[1], y=quat[2], z=quat[3])


        # print(
        #     f"In Euler(rad): \033[1;92m{math_helpers.quat_to_eulerZYX(rotation)}\033[0m"
        # )

        x, y, z = position
        offset_distance = offset # TODO: Optimize the offset
        approach_pos = geometry_pb2.Vec3(
            x=x + flipped_normal[0] * offset_distance,
            y=y + flipped_normal[1] * offset_distance,
            z=z + flipped_normal[2] * offset_distance,
        )
        print(f"\nposition: \033[1;92m({position})\033[0m")
        print(f"offset: \033[1;92m({offset})\033[0m")
        print(f"flipped_normal: \033[1;92m{flipped_normal}\033[0m")
        print(f"approach_pos: \033[1;92m{approach_pos}\033[0m")

        # grasp_pos = geometry_pb2.Vec3(
        #     x=x + flipped_normal[0],
        #     y=y + flipped_normal[1],
        #     z=z + flipped_normal[2],
        # )

        approach_vector = (
            camera_tform_body
            * math_helpers.SE3Pose.from_proto(
                geometry_pb2.SE3Pose(
                    position=geometry_pb2.Vec3(
                        x=approach_pos.x, y=approach_pos.y, z=approach_pos.z
                    ),
                    rotation=geometry_pb2.Quaternion(
                        w=rotation.w, x=rotation.x, y=rotation.y, z=rotation.z
                    ),
                )
            )
        )
        print(f"Original normal: {normal_approach_vector}")
        print(f"Euler normal: \033[1;92m{n_euler}\033[0m")
        print(
            f"rotation: \033[1;92m{rotation.w:.3f}, x={rotation.x:.3f}, y={rotation.y:.3f}, z={rotation.z:.3f}\033[0m."
        )

        # grasp_vector = (
        #     camera_tform_body
        #     * math_helpers.SE3Pose.from_proto(
        #         geometry_pb2.SE3Pose(
        #             position=geometry_pb2.Vec3(
        #                 x=grasp_pos.x, y=grasp_pos.y, z=grasp_pos.z
        #             ),
        #             rotation=geometry_pb2.Quaternion(
        #                 w=rotation.w, x=rotation.x, y=rotation.y, z=rotation.z
        #             ),
        #         )
        #     )
        # )

        if distance > 2:
            raise ValueError(
                f"Target pose (\033[1;92m{distance:.2f} m\033[0m) exceeds arm reach (~2 m)\n"
            )
        # print(
        #     f"\nFinal position: \033[1;92m(x={grasp_vector.x:.3f}, y={grasp_vector.y:.3f}, z={grasp_vector.z:.3f})\033[0m"
        # )
        # print(
        #     f"Final rotation: \033[1;92m(w={grasp_vector.rot.w:.3f}, x={grasp_vector.rot.x:.3f}, y={grasp_vector.rot.y:.3f}, z={grasp_vector.rot.z:.3f})\033[0m."
        # )
        # TODO: Crack the approach in 2 motions: 1. approach with offset+rotation 2. final approach to grasp  
        approach_approach_vector = geometry_pb2.SE3Pose( # This needs to contain the Position with Offset and the rotation
            position=geometry_pb2.Vec3(x=approach_vector.x, y=approach_vector.y, z=approach_vector.z),
            rotation=geometry_pb2.Quaternion(
                w=approach_vector.rot.w, x=approach_vector.rot.x, y=approach_vector.rot.y, z=approach_vector.rot.z
            ),
        )
        # grasp_approach_vector = geometry_pb2.SE3Pose( # this needs to contain the Position without Offset and no rotation
        #     position=geometry_pb2.Vec3(x=grasp_vector.x, y=grasp_vector.y, z=grasp_vector.z),
        #     rotation=geometry_pb2.Quaternion(
        #         w=grasp_vector.rot.w, x=grasp_vector.rot.x, y=grasp_vector.rot.y, z=grasp_vector.rot.z
        #     ),
        # )        
        command = RobotCommandBuilder.arm_pose_command_from_pose( 
            hand_pose=approach_approach_vector, frame_name=BODY_FRAME_NAME, seconds=5
        )
        # command = RobotCommandBuilder.arm_pose_command_from_pose( 
        #     hand_pose=grasp_approach_vector, frame_name="body", seconds=5
        # )        
        print(f"\napproach_vector: \033[1;92m{approach_vector}\033[0m")

        robot.logger.info("Sending gripper alignment command...")
        cmd_id = command_client.robot_command(command, timeout=98)

        robot.logger.info("Waiting for arm to reach target...")
        start_time = time.time()
        timeout = 100
        while time.time() - start_time < timeout:
            feedback_resp = command_client.robot_command_feedback(cmd_id)
            arm_feedback = (
                feedback_resp.feedback.synchronized_feedback.arm_command_feedback
            )
            if arm_feedback.HasField("arm_cartesian_feedback"):
                status = arm_feedback.arm_cartesian_feedback.status
                if (
                    status
                    == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE
                ):
                    robot.logger.info("Gripper aligned successfully.")
                    print("Gripper aligned successfully.")
                    return True
                elif status in (
                    arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_STALLED,
                    arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_CANCELLED,
                ):
                    robot.logger.warning(f"Arm trajectory failed: {status}.")
                    print(arm_feedback)
                    return True
            time.sleep(0.1)

        robot.logger.warning("Arm trajectory timed out.")
        print("Arm trajectory timed out after 7 seconds.")
        return False

    except Exception as e:
        robot.logger.error(f"Error in align_gripper: {e}. \nStowing the arm...\n")
        print(f"Error in align_gripper: {e}")
        try:
            robot_cmd = RobotCommandBuilder.arm_stow_command()
            cmd_id = command_client.robot_command(robot_cmd, timeout=5)
            success = block_until_arm_arrives(command_client, cmd_id, timeout_sec=5)
            if not success:
                robot.logger.warning("Failed to stow arm.")
        except Exception as stow_e:
            robot.logger.error(f"Failed to send stow command: {stow_e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Grasp a bottle using Spot's RGB-D, YOLO, NN, and d2nt.")
    add_base_arguments(parser)
    args = parser.parse_args()

    # Lease setup
    setup_logging(args.verbose)
    sdk = create_standard_sdk("GraspBottle")
    robot = sdk.create_robot(args.hostname)
    os.environ["BOSDYN_CLIENT_USERNAME"] = "admin"
    os.environ["BOSDYN_CLIENT_PASSWORD"] = "spotadmin2017"
    authenticate(robot)
    robot.time_sync.wait_for_sync()
    image_client = robot.ensure_client(ImageClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    
    # Load models
    MODEL_PATH = Path("models/nn-2_25-04-28_GraspNN_V3.pth")
    loaded_model = GraspNN_V3()
    loaded_model.load_state_dict(torch.load(f=MODEL_PATH)) 
    loaded_model.to(device)
    model = YOLO("yolo11n-seg.pt")
    print("YOLOv11 segmentation model loaded!")

    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.power_on(timeout_sec=20)
        if not robot.is_powered_on():
            raise Exception("Robot power on failed")
        robot.logger.info("Commanding robot to stand...")
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        print("Connected with Spot. Processing grasp...")

        # Capture images
        rgb_img, depth_img, shot, fx, fy, u0, v0 = get_rgb_depth_images(image_client, command_client)
        if rgb_img is None:
            print("Failed to capture the RGB image, exiting...")
            robot.power_off(cut_immediately=False, timeout_sec=20)
            print("Robot powered off.")
            cv2.destroyAllWindows()
            print("Disconnected.")
            return
        
        # Data preprocessing
        normal_map = d2nt(depth_img.copy(), fx, fy, u0, v0)
        new_depth: np.ndarray = np.zeros_like(depth_img)
        new_depth[:, 30:] = depth_img[:, :-30].copy()
        depth_img = new_depth
        new_normal = np.zeros_like(normal_map)
        new_normal[:, 30:, :] = normal_map[:, :-30, :].copy()
        normal_map = new_normal
        
        plot_depth_image_with_colorbar(depth_image=depth_img.copy(), name="depth_example.png")

        # Segment bottle
        seg = segment_bottle(model, rgb_img)
        if seg is None:
            print("No bottle detected, exiting...")
            return

        rgb_input, depth_input, normal_map_input, mask, dist = preprocess_data(
            rgb_img, depth_img.copy(), normal_map, seg, target=np.zeros(rgb_img.shape)
        )

        # NN grasping model
        plot_depth_image_with_colorbar(depth_image=depth_input.copy(), name="depth_input.png", max_depth = 1)
        grasp_y, grasp_x = nn_grasping_model(loaded_model, rgb_input, depth_input, mask, normal_map_input)
        grasp_pixel = grasp_y, grasp_x
        if grasp_pixel is None:
            print("No valid grasp point selected, exiting...")
            return

        # Calculate grasp parameters
        pixel, normal_approach_vector, position, distance = calculate_params(
            rgb_input, depth_img.copy(), grasp_y, grasp_x, normal_map, fx, fy, u0, v0
        )
        if position is None:
            print("No valid position or normal, exiting...")
            return
        print("Dist ", dist)
        # Align gripper and grasp
        success = align_gripper(robot, command_client, position, normal_approach_vector, -0.2, dist, shot)
        # if success:
        #     time.sleep(2)
        #     gripper_command = RobotCommandBuilder.claw_gripper_close_command(
        #         disable_force_on_contact=False, max_torque=3.0, max_vel=1.0
        #     )
        #     command_client.robot_command(gripper_command, timeout=5)
        #     print("Gripper closing.")
        #     time.sleep(5)

        # Display results
        rgb_display = rgb_img.copy()
        if pixel:
            x, y = pixel
            cv2.circle(rgb_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Gripper RGB with Grasp", rgb_display)
        # depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
        # cv2.imshow("Gripper Depth", depth_display)
        # plot_depth_image_with_colorbar(depth_image=depth_img, name="depth_grasp.png" )

        cv2.waitKey(0)  # Wait for any key to close

        robot.power_off(cut_immediately=False, timeout_sec=20)
        print("Robot powered off.")

    cv2.destroyAllWindows()
    print("Disconnected.")
    
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user; powering off robot.")
        robot = create_standard_sdk("GraspBottle").create_robot("192.168.17.2")
        robot.ensure_client(RobotCommandClient.default_service_name).robot_command(
            RobotCommandBuilder.stop_command(), timeout=2
        )
        robot.power_off(cut_immediately=False, timeout_sec=20)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)