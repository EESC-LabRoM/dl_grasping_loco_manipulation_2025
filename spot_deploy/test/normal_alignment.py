import argparse
import sys
import time
import numpy as np
import cv2
import os
import bosdyn.client
from trimesh.transformations import quaternion_from_euler
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import (
    RobotCommandClient,
    RobotCommandBuilder,
    blocking_stand,
    block_until_arm_arrives,
)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.util import setup_logging, authenticate, add_base_arguments
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME
from bosdyn.api import geometry_pb2, arm_command_pb2
from bosdyn.client import math_helpers
from scipy.spatial.transform import Rotation as R

# Global variables for mouse click
g_image_click = None
g_image_display = None

# D2NT kernels and functions (unchanged)
gradient_l = np.array([[-1, 1, 0]])
gradient_r = np.array([[0, -1, 1]])
gradient_u = np.array([[-1], [1], [0]])
gradient_d = np.array([[0], [-1], [1]])


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
    euler_angles = rotation.as_euler("xyz")

    return euler_angles

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
    # Capture images from hand cameras
    sources = ["frontleft_fisheye_image", "frontleft_depth_in_visual_frame"]
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

    reframed_depth = depth_img.T[:, ::-1]
    return (
        rgb_img,
        reframed_depth,
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
            y = int(y_rgb * depth_height / rgb_height)
            x = int(x_rgb * depth_width / rgb_width)

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

def align_gripper(robot, command_client, position, normal_vector, shot):
    try:
        # Get the Rigid Body Transformation from camera to body -> [T]
        camera_frame = "frontleft_fisheye"
        camera_tform_body = get_a_tform_b(
            shot.transforms_snapshot, BODY_FRAME_NAME, camera_frame
        )
        print(f"TRANSFORMATION INFO: {shot.transforms_snapshot}\n")

        if camera_tform_body is None:
            raise ValueError(
                f"Could not find transform from {camera_frame} to {BODY_FRAME_NAME}"
            )

        # Flip(-1*) the normal vector (point toward surface)
        flipped_normal = [-n for n in normal_vector]
        print(f"Original normal: {normal_vector}")
        print(f"Flipped normal: \033[1;92m{flipped_normal}\033[0m")

        # Create quaternion to align gripper x-axis with normal
        normal = np.array([flipped_normal[0], flipped_normal[1], flipped_normal[2]])

        quat = quaternion_from_euler(*vector_to_euler(normal))
        rotation = geometry_pb2.Quaternion(w=quat[0], x=quat[1], y=quat[2], z=quat[3])

        print(
            f"\nTarget rotation obtained by d2nt: \033[1;92mw={rotation.w:.3f}, x={rotation.x:.3f}, y={rotation.y:.3f}, z={rotation.z:.3f}\033[0m."
        )
        print(
            f"In Euler(rad): \033[1;92m{math_helpers.quat_to_eulerZYX(rotation)}\033[0m"
        )

        x, y, z = position
        print(f"\nReturned position by d2nt: {position})")
        
        # Applying the Transfomation - converting the pose from camera to body -> body_ref = [T]*_d2nt_pose_calculated
        vector = camera_tform_body * math_helpers.SE3Pose.from_proto(
            geometry_pb2.SE3Pose( # create a SE3Pose element to represent a pose with: Position(x,y,z) & Orientation(normal) got in calculate_params()
                position=geometry_pb2.Vec3(x=x, y=y, z=z),
                rotation=geometry_pb2.Quaternion(
                    w=rotation.w, x=rotation.x, y=rotation.y, z=rotation.z
                ),
            )
        )

        # Check reachability (Spot's arm reach is ~1.5–2 m from body center)
        print(
            f"\nTarget position: \033[1;92m(x={vector.x:.3f}, y={vector.y:.3f}, z={vector.z:.3f})\033[0m"
        )

        print(
            f"Target vector rotation: \033[1;92m(w={vector.rot.w:.3f}, x={vector.rot.x:.3f}, y={vector.rot.y:.3f}, z={vector.rot.z:.3f})\033[0m."
        )
        print(
            f"In Euler(rad): \033[1;92m{math_helpers.quat_to_eulerZYX(vector.rotation)}\033[0m"
        )

        distance = np.sqrt(vector.x**2 + vector.y**2 + vector.z**2)
        if distance > 2:
            raise ValueError(
                f"Target pose (\033[1;92m{distance:.2f} m\033[0m) exceeds arm reach (~2 m)"
            )

        vector = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=vector.x, y=vector.y, z=vector.z),
            rotation=geometry_pb2.Quaternion(
                w=rotation.w, x=rotation.x, y=rotation.y, z=rotation.z
            ),
        )
        command = RobotCommandBuilder.arm_pose_command_from_pose(
            hand_pose=vector, frame_name="body", seconds=5
        )
        print(f"\nArm Pose builded: \033[1;92m{vector}\033[0m")
        # Opening the gripper
        gripper_command = RobotCommandBuilder.claw_gripper_open_command()
        command_client.robot_command(gripper_command)
        # Wait 3 seconds for the gripper to fully open
        time.sleep(1.5)
        print("Gripper opened. . .")

        # Issue the command and get the command ID
        robot.logger.info("Sending gripper alignment command...")
        cmd_id = command_client.robot_command(command, timeout=5)

        # Wait for the arm to complete the trajectory
        robot.logger.info("Waiting for arm to reach target...")
        start_time = time.time()
        timeout = 7  # Give extra 2 seconds beyond trajectory duration
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
                    robot.logger.warning("Arm trajectory failed: stalled or cancelled.")
                    print("Arm trajectory failed: stalled or cancelled.")
                    return False
            time.sleep(0.1)  # Poll at 10 Hz

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
    parser = argparse.ArgumentParser(
        description="Align gripper with surface normal using D2NT and hand cameras."
    )
    add_base_arguments(parser)
    args = parser.parse_args()

    setup_logging(args.verbose)
    sdk = bosdyn.client.create_standard_sdk("NormalAlignment")
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
        robot.logger.info("Commanding robot to stand...")
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        # Capture images with gripper open
        rgb_img, depth_img, rgb_response, fx, fy, u0, v0 = get_rgb_depth_images(
            image_client, command_client
        )
        _, normal_vector, position, distance = calculate_params(
            rgb_img, depth_img, fx, fy, u0, v0
        )
        success = align_gripper(
            robot, command_client, position, normal_vector, rgb_response.shot
        )

        print(
            f"Position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) m, "
            f"Distance: {distance:.2f} m, Normal: {normal_vector}"
        )
        if not success:
            print("Alignment failed; letting Spot handle fallback.")

        robot.power_off(cut_immediately=False, timeout_sec=20)
        print("Robot powered off.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user; powering off robot.")
        robot = bosdyn.client.create_standard_sdk("NormalAlignment").create_robot(
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
