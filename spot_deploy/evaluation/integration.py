import argparse
import os
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
from trimesh.transformations import quaternion_from_euler
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
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME
from bosdyn.api import geometry_pb2, arm_command_pb2
from bosdyn.client import math_helpers

# D2NT functions from open_gripper_alignment.py
gradient_l = np.array([[-1, 1, 0]])
gradient_r = np.array([[0, -1, 1]])
gradient_u = np.array([[-1], [1], [0]])
gradient_d = np.array([[0], [-1], [1]])

def vector_to_euler(vec):
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

def vector_normalization(normal, eps=1e-8):
    mag = np.linalg.norm(normal, axis=2)
    normal /= np.expand_dims(mag, axis=2) + eps
    return normal

def soft_min(laplace_map, base=np.e, direction=0):
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

def get_rgb_depth_images(image_client, command_client):
    """Capture RGB and Depth images from Spot's gripper camera."""
    gripper_command = RobotCommandBuilder.claw_gripper_open_command()
    command_client.robot_command(gripper_command)
    time.sleep(3.0)
    print("Gripper opened to maximize camera FOV.")

    sources = ["hand_color_image", "hand_depth_in_hand_color_frame"]
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

    timestamp = time.strftime("%Y%m%d%H%M%S")
    rgb_filename = f"rgb_{timestamp}.png"
    depth_filename = f"depth_{timestamp}.npy"
    cv2.imwrite(rgb_filename, rgb_img)
    np.save(depth_filename, depth_img)
    print(f"Saved RGB image: {rgb_filename}")
    print(f"Saved Depth image: {depth_filename}")

    return rgb_img, depth_img, rgb_response.shot, fx, fy, u0, v0

def segment_bottle(model, rgb_img, depth_img):
    """Segment a bottle using YOLO and extract mask."""
    results = model(rgb_img)[0]
    bottle_class_id = 39  # COCO 'bottle' class
    bottle_found = False
    mask = None

    for box, cls, conf, m in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf, results.masks):
        if int(cls) == bottle_class_id and conf > 0.5:
            bottle_found = True
            x1, y1, x2, y2 = map(int, box)
            mask = m.data.cpu().numpy().squeeze()
            mask = (mask > 0).astype(np.uint8)
            print(f"Bottle detected at box ({x1}, {y1}, {x2}, {y2}) with confidence {conf:.3f}")
            break

    if not bottle_found:
        print("No bottle detected.")
        return None

    return mask

def nn_grasping_model(rgb_img, depth_img, mask):
    """Placeholder NN grasping model: returns a good grasp pixel."""
    if mask is None:
        return None

    # Placeholder: Pick centroid of mask
    y, x = np.where(mask)
    if len(x) == 0 or len(y) == 0:
        print("No valid pixels in mask.")
        return None
    grasp_pixel = (int(np.mean(x)), int(np.mean(y)))
    print(f"NN model selected grasp pixel: {grasp_pixel}")

    return grasp_pixel

def calculate_params(rgb_img, depth_img, grasp_pixel, fx, fy, u0, v0):
    """Calculate grasp parameters using d2nt and grasp pixel."""
    normal_map = depth_to_normal_d2nt(depth_img, fx, fy, u0, v0)
    x_rgb, y_rgb = grasp_pixel
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

    Z = depth_value / 1000.0
    if Z <= 0:
        print("Invalid depth (0 mm).")
        return None, None, None

    X = (x - u0) * Z / fx
    Y = (y - v0) * Z / fy
    distance = np.sqrt(X**2 + Y**2 + Z**2)
    normal_vector = normal_map[y, x]
    position = (X, Y, Z)

    print(f"d2nt parameters: u0={u0}, v0={v0}, fx={fx}, fy={fy}")
    print(f"Position: ({X:.3f}, {Y:.3f}, {Z:.3f}) m, Distance: {distance:.2f} m, Normal: {normal_vector}")

    return (x_rgb, y_rgb), normal_vector, position, distance

def align_gripper(robot, command_client, position, normal_vector, offset, shot):
    """Align gripper to the computed pose."""
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

        flipped_normal = [-n for n in normal_vector]
        print(f"Original normal: {normal_vector}")
        print(f"Flipped normal: \033[1;92m{flipped_normal}\033[0m")

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
        offset_distance = offset
        grasp_pos = geometry_pb2.Vec3(
            x=x + flipped_normal[0] * offset_distance,
            y=y + flipped_normal[1] * offset_distance,
            z=z + flipped_normal[2] * offset_distance,
        )
        print(f"\nReturned position by d2nt: \033[1;92m({position})\033[0m")

        vector = (
            camera_tform_body
            * math_helpers.SE3Pose.from_proto(
                geometry_pb2.SE3Pose(
                    position=geometry_pb2.Vec3(
                        x=grasp_pos.x, y=grasp_pos.y, z=grasp_pos.z
                    ),
                    rotation=geometry_pb2.Quaternion(
                        w=rotation.w, x=rotation.x, y=rotation.y, z=rotation.z
                    ),
                )
            )
        )

        distance = np.sqrt(vector.x**2 + vector.y**2 + vector.z**2)
        if distance > 2:
            raise ValueError(
                f"Target pose (\033[1;92m{distance:.2f} m\033[0m) exceeds arm reach (~2 m)\n"
            )
        print(
            f"\nTarget position: \033[1;92m(x={vector.x:.3f}, y={vector.y:.3f}, z={vector.z:.3f})\033[0m"
        )
        print(
            f"Target rotation: \033[1;92m(w={vector.rot.w:.3f}, x={vector.rot.x:.3f}, y={vector.rot.y:.3f}, z={vector.rot.z:.3f}\033[0m."
        )

        vector = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=vector.x, y=vector.y, z=vector.z),
            rotation=geometry_pb2.Quaternion(
                w=vector.rot.w, x=vector.rot.x, y=vector.rot.y, z=vector.rot.z
            ),
        )
        command = RobotCommandBuilder.arm_pose_command_from_pose(
            hand_pose=vector, frame_name="body", seconds=5
        )
        print(f"\nArm Pose built: \033[1;92m{vector}\033[0m")

        robot.logger.info("Sending gripper alignment command...")
        cmd_id = command_client.robot_command(command, timeout=5)

        robot.logger.info("Waiting for arm to reach target...")
        start_time = time.time()
        timeout = 7
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
    model = YOLO("yolo11n-seg.pt")
    print("YOLOv11 segmentation model loaded.")

    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.power_on(timeout_sec=20)
        if not robot.is_powered_on():
            raise Exception("Robot power on failed")
        robot.logger.info("Commanding robot to stand...")
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        print("Conectado ao Spot. Pressione 'q' para sair.")
        while True:
            rgb_img, depth_img, shot, fx, fy, u0, v0 = get_rgb_depth_images(image_client, command_client)
            if rgb_img is None:
                print("Falha na captura, saindo.")
                break

            mask = segment_bottle(model, rgb_img, depth_img)
            if mask is None:
                print("No bottle to grasp, saindo loop.")
                cv2.imshow("Gripper RGB", rgb_img)
                depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                cv2.imshow("Gripper Depth", depth_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            grasp_pixel = nn_grasping_model(rgb_img, depth_img, mask)
            if grasp_pixel is None:
                print("No grasp pixel selected, saindo loop.")
                cv2.imshow("Gripper RGB", rgb_img)
                depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                cv2.imshow("Gripper Depth", depth_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            pixel, normal_vector, position, distance = calculate_params(
                rgb_img, depth_img, grasp_pixel, fx, fy, u0, v0
            )
            if position is None:
                print("No valid position or normal, saindo loop.")
                cv2.imshow("Gripper RGB", rgb_img)
                depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                cv2.imshow("Gripper Depth", depth_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            success = align_gripper(robot, command_client, position, normal_vector, 0.1, shot)
            if success:
                time.sleep(2)
                gripper_command = RobotCommandBuilder.claw_gripper_close_command(
                    disable_force_on_contact=False, max_torque=3.0, max_vel=1.0
                )
                command_client.robot_command(gripper_command, timeout=5)
                print("Gripper closing.")
                time.sleep(5)

            rgb_display = rgb_img.copy()
            if pixel:
                x, y = pixel
                cv2.circle(rgb_display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Gripper RGB with Grasp", rgb_display)
            depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            cv2.imshow("Gripper Depth", depth_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        robot.power_off(cut_immediately=False, timeout_sec=20)
        print("Robot powered off.")

    cv2.destroyAllWindows()
    print("Desconectado.")

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