import argparse
import sys
import time
import numpy as np
import cv2
import os
import math
import bosdyn.client
from bosdyn import geometry
from bosdyn.client.image import ImageClient
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME
from bosdyn.client.robot_command import (
    RobotCommandClient,
    RobotCommandBuilder,
)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.util import setup_logging, authenticate, add_base_arguments

# Global variables for mouse click
g_image_click = None
g_image_display = None

# D2NT kernels and functions (unchanged)
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
        return (lap_pow_l + eps * 0.5) / (eps + lap_pow_l + lap_pow_r), (
            lap_pow_r + eps * 0.5
        ) / (eps + lap_pow_l + lap_pow_r)
    elif direction == 1:
        lap_pow_u = np.vstack([np.zeros((1, w)), lap_power[:-1, :]])
        lap_pow_d = np.vstack([lap_power[1:, :], np.zeros((1, w))])
        return (lap_pow_u + eps / 2) / (eps + lap_pow_u + lap_pow_d), (
            lap_pow_d + eps / 2
        ) / (eps + lap_pow_u + lap_pow_d)


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
        cv2.imshow("Select Point", g_image_display)


def get_rgb_depth_images(image_client, command_client):
    # Open the gripper to clear the FOV
    gripper_command = RobotCommandBuilder.claw_gripper_open_command()
    command_client.robot_command(gripper_command)

    # Wait 3 seconds for the gripper to fully open
    time.sleep(3.0)
    print("Gripper opened to maximize camera FOV.")
    # Capture images from hand cameras
    sources = ["hand_color_image", "hand_depth"]
    responses = image_client.get_image_from_sources(sources)
    if len(responses) != 2:
        raise ValueError(f"Expected 2 images, got {len(responses)}")

    for i, response in enumerate(responses):
        print(
            f"Source {i}: {response.source.name}, "
            f"Rows: {response.shot.image.rows}, Cols: {response.shot.image.cols}, "
            f"Format: {response.shot.image.pixel_format}"
        )

    rgb_img = cv2.imdecode(
        np.frombuffer(responses[0].shot.image.data, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    depth_data = np.frombuffer(responses[1].shot.image.data, dtype=np.uint16)
    depth_img = depth_data.reshape(
        responses[1].shot.image.rows, responses[1].shot.image.cols
    )

    pinhole = responses[1].source.pinhole.intrinsics
    fx, fy = pinhole.focal_length.y, pinhole.focal_length.x
    u0, v0 = pinhole.principal_point.y, pinhole.principal_point.x
    
    time.sleep(3.0)
    gripper_command = RobotCommandBuilder.claw_gripper_close_command(disable_force_on_contact=False)
    command_client.robot_command(gripper_command)

    return (
        rgb_img,
        depth_img,
        responses[0],
        fx,
        fy,
        u0,
        v0,
    )  # Return RGB response for shot


def calculate_params(rgb_img, depth_img, fx, fy, u0, v0):
    global g_image_click, g_image_display
    normal_map = depth_to_normal_d2nt(depth_img, fx, fy, u0, v0)
    depth_display = cv2.normalize(
        depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    cv2.imshow("Depth Image", depth_display)

    cv2.namedWindow("Select Point")
    cv2.setMouseCallback("Select Point", cv_mouse_callback)
    while True:
        g_image_click = None
        g_image_display = rgb_img.copy()
        while g_image_click is None:
            cv2.imshow("Select Point", g_image_display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                raise ValueError("User quit")
        x_rgb, y_rgb = g_image_click

        depth_height, depth_width = depth_img.shape
        rgb_height, rgb_width = rgb_img.shape[:2]
        print(f"Depth dimensions: {depth_height, depth_width}")
        print(f"RGB dimensions: {rgb_height, rgb_width}")

        if rgb_width == depth_width and rgb_height == depth_height:
            x = x_rgb
            y = y_rgb
        else:
            x = int(x_rgb * depth_width / rgb_width)
            y = int(y_rgb * depth_height / rgb_height)
        
        x = min(max(x, 0), depth_width - 1)
        y = min(max(y, 0), depth_height - 1)
        
        print(f"Clicked (x, y): ({x_rgb}, {y_rgb}), Depth at ({x}, {y})")
        depth_value = depth_img[y, x]
        print(
            f"Clicked (x, y): ({x_rgb}, {y_rgb}), Depth at ({x}, {y}): {depth_value} mm"
        )

        Z = depth_value / 1000.0
        if Z > 0:
            break
        print("Invalid depth (0 mm). Click again.")

    print(f"\nParâmetros do d2nt: u:{u0}, v:{v0}, fx:{fx}, fy:{fy}\n")
    X = (x - u0) * Z / fx  # Y
    Y = (y - v0) * Z / fy  # X
    distance = np.sqrt(X**2 + Y**2 + Z**2)
    normal_vector = normal_map[y, x]

    return (x_rgb, y_rgb), normal_vector, (X, Y, Z), distance

def annotate_image(rgb_img, point, normal_vector, position, distance):
    img = rgb_img.copy()
    x, y = point
    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Highlight clicked point
    
    # Draw normal direction (2D projection)
    normal_length = 30
    nx, ny = normal_vector[0], normal_vector[1]
    end_x = int(x + nx * normal_length)
    end_y = int(y + ny * normal_length)
    cv2.arrowedLine(img, (x, y), (end_x, end_y), (0, 255, 0), 2)
    
    # Add text annotations
    text = f"Pos: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) m"
    text += f"\nDist: {distance:.2f} m"
    text += f"\nNormal: ({normal_vector[0]:.2f}, {normal_vector[1]:.2f}, {normal_vector[2]:.2f})"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img

def main():
    parser = argparse.ArgumentParser(
        description="Align gripper with surface normal using D2NT and hand cameras."
    )
    add_base_arguments(parser)
    args = parser.parse_args()

    setup_logging(args.verbose)
    sdk = bosdyn.client.create_standard_sdk("ArmAnnotateRGBD2NT")
    robot = sdk.create_robot(args.hostname)
    os.environ["BOSDYN_CLIENT_USERNAME"] = "admin"
    os.environ["BOSDYN_CLIENT_PASSWORD"] = "spotadmin2017"
    authenticate(robot)
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(ImageClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)

    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.power_on(timeout_sec=20)
        if not robot.is_powered_on():
            raise Exception("Robot power on failed")

        # Capture images with gripper open
        rgb_img, depth_img, rgb_response, fx, fy, u0, v0 = get_rgb_depth_images(
            image_client, command_client
        )
        _, normal_vector, position, distance = calculate_params(
            rgb_img, depth_img, fx, fy, u0, v0
        )
        annotated_img = annotate_image(rgb_img, _, normal_vector, position, distance)
        cv2.imshow('Annotated Image', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('annotated_output.jpg', annotated_img)
        print(
            f"Position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) m, "
            f"Distance: {distance:.2f} m, Normal: {normal_vector}"
        )

        robot.power_off(cut_immediately=False, timeout_sec=20)
        print("Robot powered off.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user; powering off robot.")
        robot = bosdyn.client.create_standard_sdk("ArmAnnotateRGBD2NT").create_robot(
            "192.168.17.2"
        )
        robot.ensure_client(RobotCommandClient.default_service_name).robot_command(
            RobotCommandBuilder.stop_command(), timeout=2
        )
        robot.power_off(cut_immediately=False, timeout_sec=20)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
