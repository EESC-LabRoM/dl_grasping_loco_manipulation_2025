import argparse
import os
import sys
import time
import cv2
import numpy as np
from bosdyn.client import create_standard_sdk
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.util import add_base_arguments, setup_logging, authenticate

# Spot Cameras
# Location	   Type	    Name	                            Resolution	    Description
# Front Body	RGB	    frontleft_fisheye_image	            1280x720	    Front-left fisheye RGB camera
# Front Body	Depth	frontleft_depth	                    424x240	        Front-left depth camera
# Front Body	Depth	frontleft_depth_in_visual_frame	    1280x720	    Depth aligned with front-left RGB
# Front Body	RGB	    frontright_fisheye_image	        1280x720	    Front-right fisheye RGB camera
# Front Body	Depth	frontright_depth	                424x240	        Front-right depth camera
# Front Body	Depth	frontright_depth_in_visual_frame	1280x720	    Depth aligned with front-right RGB
# Hand (Arm)	RGB	    hand_color_image	                640x480	        Hand camera RGB
# Hand (Arm)	Depth	hand_depth	                        640x480	        Hand camera depth

def get_rgb_d_images(image_client, command_client):
    """Capture and save RGB and Depth images from Spot's gripper camera after opening gripper."""
    # Open the gripper to clear FOV
    gripper_command = RobotCommandBuilder.claw_gripper_open_command()
    command_client.robot_command(gripper_command)
    time.sleep(3.0)  # Wait for gripper to fully open
    print("Gripper opened to maximize camera FOV.")

    # Capture RGB and Depth
    sources = ["hand_color_image", "hand_depth_in_hand_color_frame"]
    responses = image_client.get_image_from_sources(sources)
    if len(responses) != 2:
        print(f"Erro: Expected 2 images, got {len(responses)}")
        return None, None

    # Process RGB
    rgb_response = responses[0]
    if rgb_response.source.name != "hand_color_image":
        print(f"Erro: Expected hand_color_image, got {rgb_response.source.name}")
        return None, None
    rgb_img = cv2.imdecode(
        np.frombuffer(rgb_response.shot.image.data, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    rgb_rows, rgb_cols = rgb_response.shot.image.rows, rgb_response.shot.image.cols
    print(
        f"RGB Source: {rgb_response.source.name}, "
        f"Resolution: {rgb_rows}x{rgb_cols}, "
        f"Shape: {rgb_img.shape}, "
        f"Format: {rgb_response.shot.image.pixel_format}"
    )

    # Process Depth
    depth_response = responses[1]
    if depth_response.source.name != "hand_depth_in_hand_color_frame":
        print(f"Erro: Expected hand_depth_in_hand_color_frame, got {depth_response.source.name}")
        return None, None
    depth_data = np.frombuffer(depth_response.shot.image.data, dtype=np.uint16)
    depth_img = depth_data.reshape(depth_response.shot.image.rows, depth_response.shot.image.cols)
    depth_rows, depth_cols = depth_response.shot.image.rows, depth_response.shot.image.cols
    print(
        f"Depth Source: {depth_response.source.name}, "
        f"Resolution: {depth_rows}x{depth_cols}, "
        f"Shape: {depth_img.shape}, "
        f"Format: {depth_response.shot.image.pixel_format}"
    )

    # Save images
    timestamp = time.strftime("%Y%m%d%H%M%S")
    rgb_filename = f"rgb_{timestamp}.png"
    depth_filename = f"depth_{timestamp}.npy"
    cv2.imwrite(rgb_filename, rgb_img)
    np.save(depth_filename, depth_img)
    print(f"Saved RGB image: {rgb_filename}")
    print(f"Saved Depth image: {depth_filename}")
    
    # Close the gripper 
    gripper_command = RobotCommandBuilder.claw_gripper_close_command()
    command_client.robot_command(gripper_command)
    time.sleep(3.0)  # Wait for gripper to fully close
    print("Gripper closed.")

    return rgb_img, depth_img

def main():
    parser = argparse.ArgumentParser(description="Extract and save RGB-D images from Spot gripper camera.")
    add_base_arguments(parser)
    args = parser.parse_args()

    setup_logging(args.verbose)
    sdk = create_standard_sdk("RGBDExtractor")
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
            # Get RGB and Depth images
            rgb_img, depth_img = get_rgb_d_images(image_client, command_client)
            if rgb_img is None or depth_img is None:
                print("Falha na captura de imagens, saindo do loop.")
                break

            # Display RGB
            cv2.imshow("Gripper RGB", rgb_img)

            # Display Depth (normalize for visualization)
            depth_display = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
            cv2.imshow("Gripper Depth", depth_display)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Power off after loop
        robot.power_off(cut_immediately=False, timeout_sec=20)
        print("Robot powered off.")

    # Cleanup
    cv2.destroyAllWindows()
    print("Desconectado.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user; powering off robot.")
        robot = create_standard_sdk("RGBDExtractor").create_robot("192.168.17.2")
        robot.ensure_client(RobotCommandClient.default_service_name).robot_command(
            RobotCommandBuilder.stop_command(), timeout=2
        )
        robot.power_off(cut_immediately=False, timeout_sec=20)
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)