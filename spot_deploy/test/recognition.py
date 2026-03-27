import argparse
import sys
import time
import numpy as np
import cv2
import os
from ultralytics import YOLO
import bosdyn.client
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import (
    RobotCommandClient,
    RobotCommandBuilder,
)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.util import setup_logging, authenticate, add_base_arguments


def recognize_objects(image_client, command_client):
    # Load YOLOv11 segmentation model
    model = YOLO("yolo11n-seg.pt")
    print("YOLOv11 segmentation model loaded.")
    # Open the gripper to clear the FOV
    gripper_command = RobotCommandBuilder.claw_gripper_open_command()
    command_client.robot_command(gripper_command)

    # Wait 3 seconds for the gripper to fully open
    time.sleep(3.0)
    print("Gripper opened to maximize camera FOV.")

    # Capture images from hand cameras
    sources = ["hand_color_image"]
    responses = image_client.get_image_from_sources(sources)
    if len(responses) != 1:
        raise ValueError(f"Expected 1 images, got {len(responses)}")

    for i, response in enumerate(responses):
        print(
            f"Source {i}: {response.source.name}, "
            f"Rows: {response.shot.image.rows}, Cols: {response.shot.image.cols}, "
            f"Format: {response.shot.image.pixel_format}"
        )

    rgb_img = cv2.imdecode(
        np.frombuffer(responses[0].shot.image.data, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    # Run YOLO segmentation
    results = model(rgb_img)
    annotated_frame = results[0].plot()  # Draw masks and boxes
    print(f"YOLO detected {len(results[0].boxes)} objects.")

    return annotated_frame, results[0]

def main():
    parser = argparse.ArgumentParser(
        description="Testing YOLO recognition model."
    )
    add_base_arguments(parser)
    args = parser.parse_args()

    setup_logging(args.verbose)
    sdk = bosdyn.client.create_standard_sdk("YOLORecognition")
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

            print("Conectado ao Spot. Pressione 'q' para sair.")
            while True:
                # Capture and recognize
                annotated_frame, yolo_results = recognize_objects(image_client, command_client)
                if annotated_frame is None:
                    print("Falha na captura de imagem, saindo do loop.")
                    break

                # Show the result
                cv2.imshow("YOLOv11 - Spot Gripper Camera", annotated_frame)

                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Power off after loop
            robot.power_off(cut_immediately=False, timeout_sec=20)
            print("Robot powered off.")
            
    time.sleep(3.0)
    gripper_command = RobotCommandBuilder.claw_gripper_close_command()
    command_client.robot_command(gripper_command)
    
    # Cleanup OpenCV windows
    cv2.destroyAllWindows()
    print("Desconectado.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user; powering off robot.")
        robot = bosdyn.client.create_standard_sdk("YOLORecognition").create_robot(
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