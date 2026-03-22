"""
This script allows the grasp process by the segmentating the depth image to identify 
the object , selecting a random point on the segmented object, computing a high-quality 
normal using the D2NT method, aligning the gripper for grasping.

-> The selection of the grasp position use segmentation and randomization. 
-> Based on the study "D2NT: A High-Performing Depth-to-Normal Translator"  :
        Uses four direction-aware gradients (DAG) with adaptive weighting, followed by 
        a more sophisticated normal construction incorporating camera intrinsics fully.
        This method also relates Pixels Coordinates (u, v) to World Coordinates (x, y, z),
        accounting for perspective distortion in the normal’s z-component.
    
    Reference: https://github.com/hongfo/depth_to_normal/tree/main

Flow:
    Captures depth → Segments image → Picks random point → Computes D2NT normal → Grasps.

"""
import argparse
import sys
import time
import os
import numpy as np
import cv2
import bosdyn.client
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.api import geometry_pb2, manipulation_api_pb2, image_pb2
from bosdyn.client.util import setup_logging, authenticate, add_base_arguments
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive

# D2NT-specific kernels
kernel_Gx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
kernel_Gy = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
gradient_l = np.array([[-1, 1, 0]])
gradient_r = np.array([[0, -1, 1]])
gradient_u = np.array([[-1], [1], [0]])
gradient_d = np.array([[0], [-1], [1]])

# Normalization: turning the vector an unitary
def vector_normalization(normal, eps=1e-8):
    mag = np.linalg.norm(normal, axis=2)
    normal /= np.expand_dims(mag, axis=2) + eps
    return normal

# filter that identtifies discontinuities and elimiantes outliers (non-coplanar points in relation to the reference point)
def get_DAG_filter(Z, base=np.e, lap_conf="1D-DLF"):
    grad_l = cv2.filter2D(Z, -1, gradient_l)
    grad_r = cv2.filter2D(Z, -1, gradient_r)
    grad_u = cv2.filter2D(Z, -1, gradient_u)
    grad_d = cv2.filter2D(Z, -1, gradient_d)
    
    if lap_conf == "1D-DLF":
        lap_hor = abs(grad_l - grad_r)
        lap_ver = abs(grad_u - grad_d)
    
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

# function to quantify the the gradient's weight and handle with discontinuities - improving normal estimator near edges 
def soft_min(laplace_map, base, direction):
    h, w = laplace_map.shape
    eps = 1e-8
    
    lap_power = np.power(base, -laplace_map)
    if direction == 0:  # horizontal
        lap_pow_l = np.hstack([np.zeros((h, 1)), lap_power[:, :-1]])
        lap_pow_r = np.hstack([lap_power[:, 1:], np.zeros((h, 1))])
        return (lap_pow_l + eps * 0.5) / (eps + lap_pow_l + lap_pow_r), \
               (lap_pow_r + eps * 0.5) / (eps + lap_pow_l + lap_pow_r)
    elif direction == 1:  # vertical
        lap_pow_u = np.vstack([np.zeros((1, w)), lap_power[:-1, :]])
        lap_pow_d = np.vstack([lap_power[1:, :], np.zeros((1, w))])
        return (lap_pow_u + eps / 2) / (eps + lap_pow_u + lap_pow_d), \
               (lap_pow_d + eps / 2) / (eps + lap_pow_u + lap_pow_d)

def depth_to_normal_d2nt(depth, fx, fy, u0, v0):
    """Compute normal map using D2NT v2 (DAG filter) approach."""
    h, w = depth.shape
    depth = depth.astype(np.float64) / 1000.0  # Convert mm to meters
    
    # Create u and v maps
    u_map = np.ones((h, 1)) * np.arange(1, w + 1) - u0  # u-u0
    v_map = np.arange(1, h + 1).reshape(h, 1) * np.ones((1, w)) - v0  # v-v0
    
    # Get depth gradients using DAG filter
    Gu, Gv = get_DAG_filter(depth)
    
    # Depth to Normal Translation
    est_nx = Gu * fx
    est_ny = Gv * fy
    est_nz = -(depth + v_map * Gv + u_map * Gu)
    est_normal = cv2.merge((est_nx, est_ny, est_nz))
    
    # Normalize vectors
    est_normal = vector_normalization(est_normal)
    
    return est_normal

def get_depth_image(image_client):
    """Capture and process depth image from the hand_depth camera."""
    image_response = image_client.get_image_from_sources(['hand_depth'])
    if len(image_response) != 1:
        raise ValueError(f"Expected 1 image, got {len(image_response)}")
    
    image = image_response[0]
    if image.shot.image.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        raise ValueError("Expected depth image format")
    
    depth_data = np.frombuffer(image.shot.image.data, dtype=np.uint16)
    depth_img = depth_data.reshape(image.shot.image.rows, image.shot.image.cols)
    pinhole = image.source.pinhole.intrinsics
    fx = pinhole.focal_length.x
    fy = pinhole.focal_length.y
    u0 = pinhole.principal_point.x
    v0 = pinhole.principal_point.y
    
    return depth_img, image, fx, fy, u0, v0

def calculate_grasp_params(depth_img, image_response, fx, fy, u0, v0):
    """Segment depth image, let user select a mask, and compute normal with D2NT."""
    # Compute normal map using D2NT
    normal_map = depth_to_normal_d2nt(depth_img, fx, fy, u0, v0)
    
    # Segment the depth image (assume objects are closer than 2000 mm)
    mask = cv2.inRange(depth_img, 1, 2000)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:
        raise ValueError("No object detected in depth image")
    
    # Prepare display image (normalize depth for visibility)
    display_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    
    # Draw numbered labels on the image
    for i in range(1, num_labels):  # Skip background (label 0)
        # Get bounding box and area for each object
        left, top, width, height, area = stats[i]
        if area < 100:  # Filter out very small noise regions
            continue
        # Draw rectangle and label number
        cv2.rectangle(display_img, (left, top), (left + width, top + height), (0, 255, 0), 2)
        cv2.putText(display_img, str(i), (left + 5, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show the image and wait for user input
    cv2.imshow('Select Object (Press 1, 2, ... to choose, q to quit)', display_img)
    print("Detected objects are numbered on the image. Press the number key (1, 2, ...) to select an object, or 'q' to quit.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise ValueError("User quit the selection process")
        if key >= ord('1') and key < ord('1') + num_labels - 1:
            selected_label = key - ord('0')
            break
    cv2.destroyAllWindows()
    
    # Create mask for the selected object
    object_mask = (labels == selected_label).astype(np.uint8)
    object_pixels = np.where(object_mask)
    if len(object_pixels[0]) == 0:
        raise ValueError("No valid pixels in selected object")
    
    # Select random point within the chosen object
    idx = np.random.randint(0, len(object_pixels[0]))
    y, x = object_pixels[0][idx], object_pixels[1][idx]
    normal_vector = normal_map[y, x]
    
    print(f"Selected object {selected_label} at random point: ({x}, {y}), Normal: {normal_vector}")
    
    return geometry_pb2.Vec2(x=x, y=y), normal_vector, image_response.shot

def create_grasp_request(pick_vec, normal_vector, shot):
    """Create grasp request with normal alignment."""
    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec,
        transforms_snapshot_for_camera=shot.transforms_snapshot,
        frame_name_image_sensor=shot.frame_name_image_sensor,
        camera_model=shot.source.pinhole
    )
    # This ensures the normal vector alignment is applied in the camera’s frame
    grasp.grasp_params.grasp_params_frame_name = shot.frame_name_image_sensor
    
    # This lets us tell Spot how to orient the gripper during the grasp
    constraint = grasp.grasp_params.allowable_orientation.add()
    # Spot’s gripper frame has x forward (along the gripper), y left, z up. This prepares gripper’s forward direction match something”
    constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
        geometry_pb2.Vec3(x=1, y=0, z=0)
    )
    # Spot will rotate the gripper so its x-axis aligns with this normal, computed in the camera frame.
    constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
        geometry_pb2.Vec3(x=float(normal_vector[0]), y=float(normal_vector[1]), z=float(normal_vector[2]))
    )
    # Sets a tolerance of 0.17 radians (~10 degrees) for the alignment
    constraint.vector_alignment_with_tolerance.threshold_radians = 0.17
    
    return manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

def execute_grasp(config):
    """Execute grasp with random position and D2NT normal alignment."""
    setup_logging(config.verbose)
    sdk = bosdyn.client.create_standard_sdk('DepthGraspRandomD2NT')
    os.environ["BOSDYN_CLIENT_USERNAME"] = "admin"
    os.environ["BOSDYN_CLIENT_PASSWORD"] = "spotadmin2017"
    robot = sdk.create_robot(config.hostname)
    authenticate(robot)
    robot.time_sync.wait_for_sync()
    
    if not robot.has_arm():
        raise Exception("Robot requires an arm")
    
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    manipulation_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    
    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.power_on(timeout_sec=20)
        if not robot.is_powered_on():
            raise Exception("Robot power on failed")
        blocking_stand(command_client, timeout_sec=10)
        print("Robot standing.")
        
        depth_img, image_response, fx, fy, u0, v0 = get_depth_image(image_client)
        pick_vec, normal_vector, shot = calculate_grasp_params(depth_img, image_response, fx, fy, u0, v0)
        grasp_request = create_grasp_request(pick_vec, normal_vector, shot)
        
        cmd_response = manipulation_client.manipulation_api_command(manipulation_api_request=grasp_request)
        
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)
            response = manipulation_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)
            state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)
            print(f"Current state: {state_name}")
            if response.current_state in [manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
                                        manipulation_api_pb2.MANIP_STATE_GRASP_FAILED]:
                break
            time.sleep(0.25)
            
        time.sleep(4.0)
        robot.power_off(cut_immediately=False, timeout_sec=20)
        print("Robot powered off.")

def main():
    parser = argparse.ArgumentParser(description="Grasp cylinder at random position with D2NT normals.")
    # parser.add_argument('--hostname', required=True, help='Robot hostname (e.g., 192.168.80.3)')
    # parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    add_base_arguments(parser)
    args = parser.parse_args()
    
    try:
        execute_grasp(args)
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == '__main__':
    if not main():
        sys.exit(1)