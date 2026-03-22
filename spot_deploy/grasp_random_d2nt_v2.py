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
from bosdyn.client.util import setup_logging, authenticate, add_base_arguments
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.api import geometry_pb2, manipulation_api_pb2, image_pb2

# D2NT-specific kernels
kernel_Gx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
kernel_Gy = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
gradient_l = np.array([[-1, 1, 0]])
gradient_r = np.array([[0, -1, 1]])
gradient_u = np.array([[-1], [1], [0]])
gradient_d = np.array([[0], [-1], [1]])

def vector_normalization(normal, eps=1e-8):
    mag = np.linalg.norm(normal, axis=2)
    normal /= np.expand_dims(mag, axis=2) + eps
    return normal

def soft_min(laplace_map, base, direction):
    h, w = laplace_map.shape
    eps = 1e-8
    lap_power = np.power(base, -laplace_map)
    if direction == 0:
        lap_pow_l = np.hstack([np.zeros((h, 1)), lap_power[:, :-1]])
        lap_pow_r = np.hstack([lap_power[:, 1:], np.zeros((h, 1))])
        return (lap_pow_l + eps * 0.5) / (eps + lap_pow_l + lap_pow_r), \
               (lap_pow_r + eps * 0.5) / (eps + lap_pow_l + lap_pow_r)
    elif direction == 1:
        lap_pow_u = np.vstack([np.zeros((1, w)), lap_power[:-1, :]])
        lap_pow_d = np.vstack([lap_power[1:, :], np.zeros((1, w))])
        return (lap_pow_u + eps / 2) / (eps + lap_pow_u + lap_pow_d), \
               (lap_pow_d + eps / 2) / (eps + lap_pow_u + lap_pow_d)

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

def depth_to_normal_d2nt(depth, fx, fy, u0, v0):
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
    return est_normal

def get_rgb_depth_images(image_client):
    sources = ['frontleft_fisheye_image', 'frontleft_depth_in_visual_frame']
    responses = image_client.get_image_from_sources(sources)
    if len(responses) != 2:
        raise ValueError(f"Expected 2 images, got {len(responses)}")
    
    for i, response in enumerate(responses):
        print(f"Source {i}: {response.source.name}, "
              f"Rows: {response.shot.image.rows}, Cols: {response.shot.image.cols}, "
              f"Format: {response.shot.image.pixel_format}")
    
    rgb_img = cv2.imdecode(np.frombuffer(responses[0].shot.image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    depth_data = np.frombuffer(responses[1].shot.image.data, dtype=np.uint16)
    depth_img = depth_data.reshape(responses[1].shot.image.rows, responses[1].shot.image.cols)
    
    pinhole = responses[1].source.pinhole.intrinsics
    fx, fy = pinhole.focal_length.x, pinhole.focal_length.y
    u0, v0 = pinhole.principal_point.x, pinhole.principal_point.y
    
    return rgb_img, depth_img, responses[0], fx, fy, u0, v0

def calculate_grasp_params(rgb_img, depth_img, rgb_response, fx, fy, u0, v0):
    """Segment depth image, let user select a mask, and compute normal with D2NT."""
    # Compute normal map using D2NT
    normal_map = depth_to_normal_d2nt(depth_img, fx, fy, u0, v0)
    
    # Segment the depth image (objects closer than 1500 mm)
    mask = cv2.inRange(depth_img, 1, 1500)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:
        raise ValueError("No object detected in depth image")
    
    # Prepare display image (use RGB for better visualization)
    display_img = rgb_img.copy()
    
    # Draw numbered labels on the RGB image
    depth_height, depth_width = depth_img.shape
    rgb_height, rgb_width = rgb_img.shape[:2]
    for i in range(1, num_labels):  # Skip background (label 0)
        left, top, width, height, area = stats[i]
        if area < 100:  # Filter small noise regions
            continue
        # Scale depth coordinates to RGB coordinates
        left_rgb = int(left * rgb_width / depth_width)
        top_rgb = int(top * rgb_height / depth_height)
        width_rgb = int(width * rgb_width / depth_width)
        height_rgb = int(height * rgb_height / depth_height)
        cv2.rectangle(display_img, (left_rgb, top_rgb), 
                     (left_rgb + width_rgb, top_rgb + height_rgb), (0, 255, 0), 2)
        cv2.putText(display_img, str(i), (left_rgb + 5, top_rgb + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show depth image for reference
    depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('Depth Image', depth_display)
    
    # Show RGB image for selection
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
    
    # Select random point within the chosen object (in depth image coordinates)
    idx = np.random.randint(0, len(object_pixels[0]))
    y_depth, x_depth = object_pixels[0][idx], object_pixels[1][idx]
    
    # Scale to RGB coordinates for grasp request
    x_rgb = int(x_depth * rgb_width / depth_width)
    y_rgb = int(y_depth * rgb_height / depth_height)
    
    # Ensure bounds (depth coordinates)
    x_depth = min(max(x_depth, 0), depth_width - 1)
    y_depth = min(max(y_depth, 0), depth_height - 1)
    
    depth_value = depth_img[y_depth, x_depth]
    print(f"Selected object {selected_label} at random point: RGB ({x_rgb}, {y_rgb}), "
          f"Depth ({x_depth}, {y_depth}), Depth value: {depth_value} mm")
    
    if depth_value <= 0:
        raise ValueError("Selected random point has invalid depth (0 mm)")
    
    normal_vector = normal_map[y_depth, x_depth]
    print(f"\nNormal vector: {normal_vector}")
    return geometry_pb2.Vec2(x=x_rgb, y=y_rgb), normal_vector, rgb_response.shot

def create_grasp_request(pick_vec, normal_vector, shot):
    # Flip the normal vector (point toward surface)
    flipped_normal = [-n for n in normal_vector]  
    print(f"Original normal: {normal_vector}")
    print(f"Flipped normal: {flipped_normal}")    
        
    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec,
        transforms_snapshot_for_camera=shot.transforms_snapshot,
        frame_name_image_sensor=shot.frame_name_image_sensor,
        camera_model=shot.source.pinhole
    )
    grasp.grasp_params.grasp_params_frame_name = shot.frame_name_image_sensor
    constraint = grasp.grasp_params.allowable_orientation.add()
    constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
        geometry_pb2.Vec3(x=1, y=0, z=0)
    )

    constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
        geometry_pb2.Vec3(x=float(flipped_normal[0]), y=float(flipped_normal[1]), z=float(flipped_normal[2]))
    )
    constraint.vector_alignment_with_tolerance.threshold_radians = 0.17
    return manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

def execute_grasp(config):
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
        
        rgb_img, depth_img, rgb_response, fx, fy, u0, v0 = get_rgb_depth_images(image_client)
        pick_vec, normal_vector, shot = calculate_grasp_params(rgb_img, depth_img, rgb_response, fx, fy, u0, v0)
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
    parser = argparse.ArgumentParser(description="Grasp object at random position with D2NT normals.")
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