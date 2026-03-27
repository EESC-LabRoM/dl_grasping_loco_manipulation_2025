#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import cv2
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, BODY_FRAME_NAME
from google.protobuf import wrappers_pb2
from ultralytics import YOLO  # :contentReference[oaicite:7]{index=7}
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2

from bosdyn.client import (
    create_standard_sdk,
    util,
)  # :contentReference[oaicite:9]{index=9}
from bosdyn.client.image import (
    ImageClient,
    build_image_request,
    pixel_to_camera_space,
)  # :contentReference[oaicite:10]{index=10}
from bosdyn.client.manipulation_api_client import (
    ManipulationApiClient,
)  # :contentReference[oaicite:11]{index=11}
from bosdyn.client.lease import (
    LeaseClient,
    LeaseKeepAlive,
    ResourceAlreadyClaimedError,
)  # :contentReference[oaicite:13]{index=13}
from bosdyn.client.robot_command import (
    RobotCommandClient,
    RobotCommandBuilder,
    block_until_arm_arrives,
)  # :contentReference[oaicite:14]{index=14}
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
print("PyTorch version: ", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("PyTorch-Cuda version: ", torch.version.cuda)
print("Device name: ", torch.cuda.get_device_name(0))
device = "cuda" if torch.cuda.is_available() else "cpu"

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.cm as cm


def save_image_with_colorbar(
    depth_image, name, window_title="Gripper Depth", num_bins=12, max_depth=1000
):
    min_depth = 0
    # Clip values to the defined range for better visualization
    clipped_depth = np.clip(depth_image, min_depth, max_depth)
    bin_edges = np.linspace(min_depth, max_depth, 13)
    binned_depth_indices = np.digitize(clipped_depth, bin_edges, right=False)
    binned_depth_indices = np.clip(binned_depth_indices - 1, 0, 11)
    base_cmap = cm.get_cmap(
        "viridis", 12
    )  # Get a colormap discretized to num_bins colors
    norm = Normalize(vmin=0, vmax=11)

    # --- Matplotlib Plotting ---
    plt.figure(figsize=(20, 16))
    img = plt.imshow(
        binned_depth_indices,
        cmap=base_cmap,
        norm=norm,
        origin="upper",
        extent=[0, depth_image.shape[1], depth_image.shape[0], 0],
    )
    cbar = plt.colorbar(
        img,
        ticks=np.arange(num_bins) + 0.5,
        orientation="vertical",
        label=f"Depth (mm)",
    )

    # Set the labels for the colorbar ticks to show the actual depth ranges
    cbar_labels = []
    for i in range(num_bins):
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i + 1]
        cbar_labels.append(f"{lower_bound:.2f} - {upper_bound:.2f}")

    cbar.ax.set_yticklabels(cbar_labels)
    cbar.ax.tick_params(labelsize=8)

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
            nn.GroupNorm(8, out_c),
            nn.ReLU(inplace=True),
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

        self.enc1 = self.backbone[:3]  # 24 ch
        self.enc2 = self.backbone[3:6]  # 32 ch
        self.enc3 = self.backbone[6:10]  # 64 ch   (skip)
        self.enc4 = self.backbone[10:]  # 1280 ch (bottleneck)

        self.up3 = UpBlock(1280, 64, base * 8)  # <-- canais certos
        self.up2 = UpBlock(base * 8, 32, base * 4)
        self.up1 = UpBlock(base * 4, 24, base * 2)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 2, stride=2, bias=False),
            nn.GroupNorm(8, base),
            nn.ReLU(inplace=True),
        )
        self.outc = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x = self.enc4(s3)

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
    print(
        "\n\033[1;92mGetting RGB and Depth . . .\033[0m\n"
        "Gripper opened to maximize camera FOV."
    )

    sources = [
        "hand_color_image",
        "hand_depth_in_hand_color_frame",
    ]  # hand_depth_in_hand_color_frame
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
        print(
            f"Erro: Expected hand_depth_in_hand_color_frame, got {depth_response.source.name}"
        )
        return None, None, None, None, None, None
    depth_data = np.frombuffer(depth_response.shot.image.data, dtype=np.uint16)
    depth_img = depth_data.reshape(
        depth_response.shot.image.rows, depth_response.shot.image.cols
    )
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

    annotated_frame = results.plot()

    for i, (cls, conf, m) in enumerate(
        zip(results.boxes.cls, results.boxes.conf, results.masks)
    ):
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
    cv2.imshow("recognition", annotated_frame)  # Scale for visibility
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
    mask = cv2.inRange(
        mask, 1, 1
    )  # Select only the object, removing the gripper or other elements
    mask = cv2.erode(mask, kernel, iterations=3).astype(int)
    print(f"    2. Depth step")
    # TODO: review
    depth[mask == 0] = 2000  # Place everything that is not the object 2 meters away
    depth = depth / depth.max()  # Normalizes depth

    print(f"    3. Normal step")
    normal[mask == 0] = (
        np.array([0.49803922, 0.49803922, 1.0]) * 255
    )  # Removes unwanted normals

    # grasp = (cv2.inRange(grasp, 0, 0).astype(bool) * mask).astype(int) + grasp # Preenche os espaços vazios da garrafa

    target = cv2.dilate(target, kernel, iterations=2)

    return rgb, depth, normal, mask, dist


def nn_grasping_model(loaded_model, rgb_img, depth_img, mask, normal_map):
    """Here the data will be passed through the loaded_model"""
    print(f"\n\033[1;92mPassing data through the loaded model . . .\033[0m")
    view_rgb_t = torch.from_numpy(np.copy(rgb_img)).permute(2, 0, 1).float()
    view_depth_t = (
        torch.from_numpy(np.copy(depth_img)).unsqueeze(dim=2).permute(2, 0, 1).float()
    )
    view_normal_t = torch.from_numpy(np.copy(normal_map)).permute(2, 0, 1).float()
    view_seg_t = (
        torch.from_numpy(np.copy(mask)).unsqueeze(dim=2).permute(2, 0, 1).float()
    )
    data = (
        torch.cat(
            [
                view_rgb_t,
                view_depth_t,
                view_normal_t,
                view_seg_t,
            ],
            dim=0,
        )
        .unsqueeze(dim=0)
        .to(device)
    )

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


def calculate_params(
    rgb_img, depth_img, grasp_y, grasp_x, normal_map, fx, fy, u0, v0, depth_value
):
    """Calculate grasp parameters using d2nt and grasp pixel."""
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
    # depth_value = dist #depth_img[y, x]
    print(
        f"Grasp pixel (x, y): ({x_rgb}, {y_rgb}), Depth at ({x}, {y}): {depth_value} mm"
    )
    # if (depth_value > 800) or (depth_value < 100):
    #     raise ValueError("Erro no sensor de distância")

    Z = depth_value  # / 1000.0  # Convert to meters for position calculation
    if Z <= 0:
        print("Invalid depth (0 mm).")
        return None, None, None, None

    X = (x - u0) * Z / fx
    Y = (y - v0) * Z / fy
    distance = np.sqrt(X**2 + Y**2 + Z**2)
    normal_approach_vector = normal_map[y, x]
    position = (X, Y, Z)

    print(f"d2nt parameters: u0={u0}, v0={v0}, fx={fx}, fy={fy}")
    print(
        f"Position: ({X:.3f}, {Y:.3f}, {Z:.3f}) m, Distance: {distance:.2f} m, Normal: {normal_approach_vector}"
    )

    return (x_rgb, y_rgb), normal_approach_vector, position, distance


def run_command_synchronously(command_client, command, timeout=15): ...


def calculate_approach_vector(normal_approach_vector, offset, position, shot):
    hand_frame = "hand_color_image_sensor"
    camera_tform_body = get_a_tform_b(
        shot.transforms_snapshot, BODY_FRAME_NAME, hand_frame
    )

    if camera_tform_body is None:
        raise ValueError(
            f"Could not find transform from {hand_frame} to {BODY_FRAME_NAME}"
        )

    # normal_approach_vector = np.array([0, 0, -1])
    flipped_normal = np.array(
        [
            -normal_approach_vector[0],
            -normal_approach_vector[1],
            -normal_approach_vector[2],
        ]
    )
    normal_euler = approach_vector_to_euler(flipped_normal)
    normal_quat = quaternion_from_euler(*normal_euler)

    x, y, z = position

    approach_pose = geometry_pb2.SE3Pose(
        position=geometry_pb2.Vec3(
            x=x + flipped_normal[0] * offset,
            y=y + flipped_normal[1] * offset,
            z=z + flipped_normal[2] * offset,
        ),
        rotation=geometry_pb2.Quaternion(
            w=normal_quat[0], x=normal_quat[1], y=normal_quat[2], z=normal_quat[3]
        ),
    )
    approach_vector = camera_tform_body * math_helpers.SE3Pose.from_proto(approach_pose)
    print(f"\nposition: \033[1;92m({position})\033[0m")
    print(f"offset: \033[1;92m({offset})\033[0m")
    print(f"flipped_normal: \033[1;92m{flipped_normal}\033[0m")
    print(f"Approach pose: {approach_pose}")
    print(f"\nApproach_vector: \033[1;92m{approach_vector}\033[0m")
    return approach_vector


def align_gripper(robot, command_client, approach_vector):
    """Align gripper to the computed pose."""
    print("\n\033[1;92mAligning gripper. . .\033[0m")
    # TODO: Crack the approach in 2 motions: 1. approach with offset+rotation 2. final approach to grasp
    command = RobotCommandBuilder.arm_pose_command_from_pose(
        hand_pose=approach_vector.to_proto(), frame_name=BODY_FRAME_NAME, seconds=5
    )

    robot.logger.info("Sending gripper alignment command...")
    cmd_id = command_client.robot_command(command, timeout=15)
    return block_until_arm_arrives(command_client, cmd_id, timeout_sec=15)


def loop_until_detect_bottle(model, image_client, command_client):
    captured_image = False
    while not captured_image:
        rgb_img, depth_img, shot, fx, fy, u0, v0 = get_rgb_depth_images(
            image_client, command_client
        )

        if rgb_img is None:
            print("Failed to capture the RGB image, trying again...")
            continue

        # Segment bottle
        seg = segment_bottle(model, rgb_img)

        if seg is None:
            print("No bottle detected, trying again...")
            continue

        captured_image = True
        return rgb_img, depth_img, seg, shot, fx, fy, u0, v0


def walk_and_gaze(img_client, cmd_client, manip_client):
    # 5. Capture & YOLO detect bottle
    model = YOLO("yolo11m-seg.pt")

    while True:
        resp_color, resp_depth = img_client.get_image(
            [
                build_image_request("hand_color_image", quality_percent=90),
                build_image_request("hand_depth_in_hand_color_frame"),
            ]
        )
        color_img = cv2.imdecode(
            np.frombuffer(resp_color.shot.image.data, np.uint8), cv2.IMREAD_COLOR
        )
        depth_raw = np.frombuffer(resp_depth.shot.image.data, dtype=np.uint16).reshape(
            resp_depth.shot.image.rows, resp_depth.shot.image.cols
        )
        depth_map = depth_raw * resp_depth.source.depth_scale

        results = model.predict(source=color_img, max_det=1000)[
            0
        ]  # :contentReference[oaicite:17]{index=17}

        if (
            results is None
            or results.boxes.cls is None
            or results.boxes.conf is None
            or results.masks is None
        ):
            cv2.imshow("recognition", color_img)
            print("No results")
            time.sleep(1)
            continue
        boxes, confs, cls_ids = (
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.conf.cpu().numpy(),
            results.boxes.cls.cpu().numpy().astype(int),
        )

        for i, (cls, conf, m) in enumerate(
            zip(results.boxes.cls, results.boxes.conf, results.masks)
        ):
            class_name = results.names[int(cls)]
            print(class_name)
            if class_name in ["bottle", "vase"]:
                print(f"Detected {class_name} ({cls})with confidence {conf:.3f}")
                mask = m.data.cpu().numpy().squeeze()  # Get binary mask
                mask = (mask > 0).astype(np.uint8)  # Ensure binary (0 or 1)

        idxs = np.where((cls_ids == 39) & (confs > 0.27))[0]

        cv2.imshow("recognition", results.plot())  # Scale for visibility
        cv2.imwrite("recognition.png", results.plot())
        if not idxs.size:
            print("No bottle detected, retrying...")
            cv2.waitKey(1000)
        else:
            print(f"Detected {len(idxs)} bottles, using the first one.")
            # cv2.waitKey(0)
            break

    x1, y1, x2, y2 = boxes[idxs[0]]
    u_rgb, v_rgb = int((x1 + x2) / 2), int((y1 + y2) / 2)

    # 6. Walk to the YOLO center pixel

    walk_vec = geometry_pb2.Vec2(
        x=u_rgb, y=v_rgb
    )  # :contentReference[oaicite:19]{index=19}
    walk_to = manipulation_api_pb2.WalkToObjectInImage(
        pixel_xy=walk_vec,
        transforms_snapshot_for_camera=resp_color.shot.transforms_snapshot,
        frame_name_image_sensor=resp_color.shot.frame_name_image_sensor,
        camera_model=resp_color.source.pinhole,
        offset_distance=wrappers_pb2.FloatValue(value=1),
    )

    manip_req = manipulation_api_pb2.ManipulationApiRequest(
        walk_to_object_in_image=walk_to
    )

    resp = manip_client.manipulation_api_command(
        manipulation_api_request=manip_req
    )  # :contentReference[oaicite:20]{index=20}
    while True:
        time.sleep(0.25)
        fb_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=resp.manipulation_cmd_id
        )
        feedback = manip_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=fb_req
        )
        if feedback.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
            break  # :contentReference[oaicite:21]{index=21}

    gaze_command = RobotCommandBuilder.arm_gaze_command(
        1.2,
        0,
        0.0,
        BODY_FRAME_NAME,
        max_linear_vel=0.1,
        max_angular_vel=0.1,
        max_accel=0.1,
    )
    gaze_command_id = cmd_client.robot_command(gaze_command)
    block_until_arm_arrives(cmd_client, gaze_command_id, 20.0)


def main():
    # parser = argparse.ArgumentParser(
    #     description="Grasp a bottle using Spot's RGB-D, YOLO, NN, and d2nt."
    # )
    # add_base_arguments(parser)
    # args = parser.parse_args()

    # Lease setup
    setup_logging(False)
    sdk = create_standard_sdk("GraspBottle")
    robot = sdk.create_robot("192.168.17.2")
    os.environ["BOSDYN_CLIENT_USERNAME"] = "admin"
    os.environ["BOSDYN_CLIENT_PASSWORD"] = "spotadmin2017"
    authenticate(robot)
    robot.time_sync.wait_for_sync()
    image_client = robot.ensure_client(ImageClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    manip_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    # Load models
    MODEL_PATH = Path("models/nn-2_25-04-28_GraspNN_V3.pth")
    loaded_model = GraspNN_V3()
    loaded_model.load_state_dict(torch.load(f=MODEL_PATH))
    loaded_model.to(device)
    model = YOLO("yolo11m-seg.pt")
    print("YOLOv11 segmentation model loaded!")

    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.power_on(timeout_sec=20)
        if not robot.is_powered_on():
            raise Exception("Robot power on failed")
        robot.logger.info("Commanding robot to stand...")
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        print("Connected with Spot. Processing grasp...")

        walk_and_gaze(image_client, command_client, manip_client)

        # Capture images
        rgb_img, depth_img, seg, shot, fx, fy, u0, v0 = loop_until_detect_bottle(
            model,
            image_client,
            command_client,
        )

        # Shift depth and normal matrices to match the RGB image
        normal_map = d2nt(depth_img.copy(), fx, fy, u0, v0)
        new_depth: np.ndarray = np.zeros_like(depth_img)
        new_depth[:, 30:] = depth_img[:, :-30].copy()
        depth_img = new_depth
        new_normal = np.zeros_like(normal_map)
        new_normal[:, 30:, :] = normal_map[:, :-30, :].copy()
        normal_map = new_normal
        save_image_with_colorbar(depth_image=depth_img.copy(), name="depth_example.png")

        rgb_input, depth_input, normal_map_input, mask, dist = preprocess_data(
            rgb_img, depth_img.copy(), normal_map, seg, target=np.zeros(rgb_img.shape)
        )

        # NN grasping model
        save_image_with_colorbar(
            depth_image=depth_input.copy(), name="depth_input.png", max_depth=1
        )
        grasp_y, grasp_x = nn_grasping_model(
            loaded_model, rgb_input, depth_input, mask, normal_map_input
        )
        grasp_pixel = grasp_y, grasp_x
        if grasp_pixel is None:
            print("No valid grasp point selected, exiting...")
            return

        # Calculate grasp parameters
        pixel, normal_approach_vector, position, distance = calculate_params(
            rgb_input,
            depth_img.copy(),
            grasp_y,
            grasp_x,
            normal_map,
            fx,
            fy,
            u0,
            v0,
            dist,
        )
        if position is None:
            print("No valid position or normal, exiting...")
            return

        # Display results
        rgb_display = rgb_img.copy()
        if pixel:
            x, y = pixel
            cv2.circle(rgb_display, (x, y), 5, (0, 255, 0), -1)

        # cv2.imshow("Gripper RGB with Grasp", rgb_display)
        cv2.imwrite("grasp_result.png", rgb_display)
        # cv2.waitKey(0)  # Wait for any key to close

        print("Dist ", dist)
        print("Distance ", distance)
        if distance > 1.9:
            raise ValueError(
                f"Target pose (\033[1;92m{distance:.2f} m\033[0m) exceeds arm reach (~2 m)\n"
            )
        # Align gripper and grasp
        target_pose = calculate_approach_vector(
            normal_approach_vector, -0.03, position, shot
        )
        success = align_gripper(robot, command_client, target_pose)

        time.sleep(2)
        gripper_command = RobotCommandBuilder.claw_gripper_close_command(
            disable_force_on_contact=False, max_torque=3.0, max_vel=1.0
        )
        command_client.robot_command(gripper_command, timeout=5)
        robot.logger.info("Gripper closing...")
        time.sleep(10)
        stow_command = RobotCommandBuilder.arm_stow_command()
        command_client.robot_command(stow_command, timeout=5)
        robot.logger.info("Stow")
        time.sleep(10)

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
