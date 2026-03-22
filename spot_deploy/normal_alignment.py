import argparse
import sys
import numpy as np
import cv2
import os
import bosdyn.client
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.frame_helpers import get_vision_tform_body, BODY_FRAME_NAME, VISION_FRAME_NAME
from bosdyn.client.util import setup_logging, authenticate, add_base_arguments
from bosdyn.api import geometry_pb2, image_pb2, arm_command_pb2

# Global variables for mouse click
g_image_click = None
g_image_display = None

# D2NT kernels and functions
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

def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    if event == cv2.EVENT_LBUTTONDOWN:
        g_image_click = (x, y)
        cv2.circle(g_image_display, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Select Point', g_image_display)

def get_depth_image(image_client):
    responses = image_client.get_image_from_sources(['hand_depth'])
    if len(responses) != 1:
        raise ValueError(f"Expected 1 image, got {len(responses)}")
    image = responses[0]
    if image.shot.image.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        raise ValueError("Expected depth image format")
    depth_data = np.frombuffer(image.shot.image.data, dtype=np.uint16)
    depth_img = depth_data.reshape(image.shot.image.rows, image.shot.image.cols)
    pinhole = image.source.pinhole.intrinsics
    fx, fy = pinhole.focal_length.x, pinhole.focal_length.y
    u0, v0 = pinhole.principal_point.x, pinhole.principal_point.y
    return depth_img, image, fx, fy, u0, v0

def calculate_params(depth_img, image_response, fx, fy, u0, v0):
    global g_image_click, g_image_display
    g_image_click = None
    g_image_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    g_image_display = cv2.cvtColor(g_image_display, cv2.COLOR_GRAY2BGR)
    
    normal_map = depth_to_normal_d2nt(depth_img, fx, fy, u0, v0)
    
    cv2.namedWindow('Select Point')
    cv2.setMouseCallback('Select Point', cv_mouse_callback)
    while g_image_click is None:
        cv2.imshow('Select Point', g_image_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            raise ValueError("User quit")
    x, y = g_image_click
    
    # Compute 3D position in camera frame
    Z = depth_img[y, x] / 1000.0  # Depth in meters
    if Z == 0:
        raise ValueError("Invalid depth at clicked point (0 mm)")
    X = (x - u0) * Z / fx
    Y = (y - v0) * Z / fy
    normal_vector = normal_map[y, x]
    
    return (X, Y, Z), normal_vector, image_response.shot

def align_gripper(robot, command_client, position, normal_vector, shot):
    # Convert position from camera frame to vision frame
    vision_tform_body = get_vision_tform_body(shot.transforms_snapshot)
    camera_tform_body = shot.transforms_snapshot.child_to_parent.edge_tform_child.inverse()
    camera_pos = geometry_pb2.Vec3(x=position[0], y=position[1], z=position[2])
    body_pos = camera_tform_body * camera_pos
    
    # Position the gripper 10 cm away along the normal
    offset_distance = 0.1  # 10 cm in meters
    grasp_pos = geometry_pb2.Vec3(
        x=body_pos.x - normal_vector[0] * offset_distance,
        y=body_pos.y - normal_vector[1] * offset_distance,
        z=body_pos.z - normal_vector[2] * offset_distance
    )
    
    # Create quaternion to align gripper x-axis with normal
    normal = np.array([normal_vector[0], normal_vector[1], normal_vector[2]])
    x_axis = np.array([1, 0, 0])  # Gripper x-axis
    if np.allclose(normal, x_axis):
        rotation = geometry_pb2.Quaternion(w=1, x=0, y=0, z=0)  # Identity
    else:
        # Compute rotation from x-axis to normal using axis-angle
        cross = np.cross(x_axis, normal)
        dot = np.dot(x_axis, normal)
        angle = np.arccos(dot)
        axis = cross / np.linalg.norm(cross)
        half_angle = angle / 2
        sin_half = np.sin(half_angle)
        rotation = geometry_pb2.Quaternion(
            w=np.cos(half_angle),
            x=axis[0] * sin_half,
            y=axis[1] * sin_half,
            z=axis[2] * sin_half
        )
    
    # Build the arm pose command
    pose = geometry_pb2.SE3Pose(position=grasp_pos, rotation=rotation)
    arm_command = arm_command_pb2.ArmCartesianCommand.Request(
        root_frame_name=BODY_FRAME_NAME,
        pose_in_root_frame=pose,
        pose_trajectory_in_task=geometry_pb2.SE3Trajectory(
            points=[geometry_pb2.SE3TrajectoryPoint(pose=pose)]
        )
    )
    command = RobotCommandBuilder.synchro_command(
        command=arm_command_pb2.ArmCommand.Request(cartesian_command=arm_command)
    )
    
    # Execute the command
    command_client.robot_command(command, timeout_sec=10)
    print("Gripper aligned with normal.")

def main():
    parser = argparse.ArgumentParser(description="Align gripper with normal direction using D2NT.")
    add_base_arguments(parser)
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    sdk = bosdyn.client.create_standard_sdk('AlignGripperWithNormal')
    os.environ["BOSDYN_CLIENT_USERNAME"] = "admin"
    os.environ["BOSDYN_CLIENT_PASSWORD"] = "spotadmin2017"
    robot = sdk.create_robot(args.hostname)
    authenticate(robot)
    robot.time_sync.wait_for_sync()
    
    if not robot.has_arm():
        raise Exception("Robot requires an arm")
    
    image_client = robot.ensure_client(ImageClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    
    robot.power_on(timeout_sec=20)
    if not robot.is_powered_on():
        raise Exception("Robot power on failed")
    blocking_stand(command_client, timeout_sec=10)
    print("Robot standing.")
    
    depth_img, image_response, fx, fy, u0, v0 = get_depth_image(image_client)
    position, normal_vector, shot = calculate_params(depth_img, image_response, fx, fy, u0, v0)
    align_gripper(robot, command_client, position, normal_vector, shot)
    
    robot.power_off(cut_immediately=False, timeout_sec=20)
    print("Robot powered off.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)