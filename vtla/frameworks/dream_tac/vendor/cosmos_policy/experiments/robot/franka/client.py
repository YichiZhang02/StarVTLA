import base64
import os
import time
import datetime

import cv2
import numpy as np
import pyrealsense2 as rs
import requests
from scipy.spatial.transform import Rotation as R

# --- 1. CONFIGURATION ---
#
# Checklist when client works worse than open-loop (run_franka_openloop.py):
#   1. Image color: RealSense gives BGR; server/cosmos_utils expect RGB. We must send RGB (convert before encode).
#   2. Instruction: must match T5 cache key exactly (e.g. "pickandplacebaguettefromplatetobasket" from preprocessing).
#   3. State (proprio): 6-dim [x, y, z, roll, pitch, yaw] pose only, same units as training.
#   4. Action: 7-dim (pose 6 + gripper 1); action[6] is gripper in dataset scale.
#   5. Camera order: cam_0 = front (primary), cam_1 = wrist (cam_high). Swap if your hardware is reversed.
#   6. Action execution: use full chunk [0..chunk_size-1] unless you intentionally skip warm-up steps.

# URL of the inference server
INFERENCE_SERVER_URL = "http://172.20.10.2:8000/infer"

# URL of the local Franka ROS control server
FRANKA_CONTROL_SERVER_URL = "http://127.0.0.1:5000"

# Directory to save periodic image logs
LOG_IMAGE_DIR = "./log_images_franka"

# --- 2. HARDWARE & HELPER CLASSES/FUNCTIONS ---

def euler_to_quat(euler_angles: np.ndarray, convention: str = 'xyz') -> np.ndarray:
    """
    将一批欧拉角转换为四元数。

    Args:
        euler_angles (np.ndarray): 形状为 (N, 3) 或 (3,) 的欧拉角。
        convention (str): 欧拉角旋转顺序。

    Returns:
        np.ndarray: 形状为 (N, 4) 或 (4,) 的四元数 (xyzw)。
    """
    rot = R.from_euler(convention, euler_angles)
    return rot.as_quat()



class MultiCameraRecorder:
    """A class to manage multiple RealSense cameras."""
    def __init__(self, serials):
        self.serials = serials
        self.pipelines = []
        self.profiles = []
        self.align = rs.align(rs.stream.color)

        for serial in self.serials:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = pipeline.start(config)
            self.pipelines.append(pipeline)
            self.profiles.append(profile)
            print(f"Started camera with serial: {serial}")

    def record_frames(self) -> dict:
        """Capture frames from all connected cameras."""
        frames_dict = {}
        for i, pipeline in enumerate(self.pipelines):
            try:
                frames = pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                if color_frame:
                    frames_dict[f'cam_{i}'] = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
            except RuntimeError as e:
                print(f"Error capturing frame from camera {i}: {e}")
                frames_dict[f'cam_{i}'] = None
        return frames_dict

    def stop(self):
        """Stops all camera pipelines."""
        for pipeline in self.pipelines:
            pipeline.stop()

def get_franka_state():
    """Retrieves state as 7-dim [x, y, z, roll, pitch, yaw, gripper] for policy (same format as action)."""
    pose_res = requests.post(f"{FRANKA_CONTROL_SERVER_URL}/getpos")
    pose_quat = pose_res.json()["pose"]  # [x, y, z, qx, qy, qz, qw]
    xyz = pose_quat[:3]
    quat = pose_quat[3:7]
    euler = R.from_quat(quat).as_euler("xyz")
    pose_6 = list(xyz) + list(euler)  # [x, y, z, roll, pitch, yaw]

    gripper_res = requests.post(f"{FRANKA_CONTROL_SERVER_URL}/get_gripper")
    gripper_width = gripper_res.json()["gripper"]

    return pose_6 + [gripper_width]  # 7-dim

def control_franka_pose(pose, gripper_state):
    """Sends pose and gripper commands to the control server."""
    # Send pose command
    if len(pose) == 6:
        # Convert from [x, y, z, roll, pitch, yaw] to [x, y, z, qx, qy, qz, qw]
        euler = np.array(pose[3:6], dtype=np.float32)
        quat = euler_to_quat(euler.reshape(1, 3)).flatten() # (4,)
        pose = pose[0:3] + quat.tolist()
    requests.post(f"{FRANKA_CONTROL_SERVER_URL}/pose", json={"arr": pose})
    
    # Send gripper command (action[6] from model is in dataset scale ~[0.08, 1.02], not 0-100)
    GRIPPER_CLOSE_THRESHOLD = 0.5
    if gripper_state > GRIPPER_CLOSE_THRESHOLD:
        requests.post(f"{FRANKA_CONTROL_SERVER_URL}/close_gripper")
    else:
        requests.post(f"{FRANKA_CONTROL_SERVER_URL}/open_gripper")

def encode_image(img: np.ndarray) -> str:
    """Encodes an image into a base64 PNG string. Expects BGR (OpenCV/RealSense); converts to RGB for server."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.png', rgb)
    return base64.b64encode(buffer).decode('utf-8')

def setup_logging_directory():
    """Creates the logging directory if it doesn't exist."""
    if not os.path.exists(LOG_IMAGE_DIR):
        os.makedirs(LOG_IMAGE_DIR)
        print(f"Created log directory: {LOG_IMAGE_DIR}")

# --- 3. MAIN EXECUTION ---

def main():
    """Main function to connect to the robot, run the control loop, and handle shutdown."""
    setup_logging_directory()
    cameras = None

    try:
        # --- A. INITIALIZATION ---
        print("Initializing cameras...")
        # Find connected RealSense devices
        connected_devices = [d.get_info(rs.camera_info.serial_number) for d in rs.context().devices]
        if len(connected_devices) < 2:
            raise RuntimeError("Expected at least 2 RealSense cameras, but found fewer.")
        # Assign cameras based on a fixed order (e.g., by serial number)
        # IMPORTANT: You may need to adjust this order based on your setup.
        # Let's assume the first is 'front' and the second is 'wrist'.
        camera_serials = sorted(connected_devices) 
        cameras = MultiCameraRecorder(camera_serials)
        print(f"Initialized cameras: Front={camera_serials[0]}, Wrist={camera_serials[1]}")
        
        print("Moving robot to initial pose...")
        # initial_pose = [0.305, 0.0, 0.481, 1.0, 0.0, 0.0, 0.0]
        initial_pose = [0.42090286014076733,-0.02565252825001829,0.20012076667459776,3.1415926*2-3.1004332835334636,0.04282809232716156,-0.10633952783455136]
        control_franka_pose(initial_pose, 0)  # Pose and open gripper (0 = open, 1 = close in model scale)
        time.sleep(1)
        print("Initialization complete!")

        # --- B. MAIN CONTROL LOOP ---
        print("Entering main control loop...")
        while True:
            # --- i. Get Observations ---
            print("\n" + "="*50)
            print("1. Gathering robot state and images...")
            
            # Get robot state
            state = get_franka_state()
            print(f"Current robot state: {np.round(state, 3)}")

            # Get images
            frames = cameras.record_frames()
            front_image = frames.get('cam_0')
            wrist_image = frames.get('cam_1')

            if front_image is None or wrist_image is None:
                print("Warning: Failed to capture one or more images. Skipping cycle.")
                time.sleep(1)
                continue

            # --- ii. Prepare Data for Server ---
            print("2. Preparing data for inference server...")
            encoded_front = encode_image(front_image)
            encoded_wrist = encode_image(wrist_image)

            # Log current images
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cv2.imwrite(os.path.join(LOG_IMAGE_DIR, f"front_{timestamp}.png"), front_image)
            cv2.imwrite(os.path.join(LOG_IMAGE_DIR, f"wrist_{timestamp}.png"), wrist_image)
            print(state)

            request_data = {
                "images": {
                    "cam_front": encoded_front,
                    "cam_high": encoded_wrist,
                },
                "state": state,
                "instruction": "pickandplacebaguettefromplatetobasket",
            }
            
            # --- iii. Send Request to Server ---
            print(f"3. Sending request to {INFERENCE_SERVER_URL}...")
            try:
                response = requests.post(INFERENCE_SERVER_URL, json=request_data, timeout=60)
                response.raise_for_status()
                result = response.json()
                print("...Success! Received response from server.")
            except requests.exceptions.RequestException as e:
                print(f"Error communicating with server: {e}. Retrying after 5s.")
                time.sleep(5)
                continue
                
            # --- iv. Parse and Execute Actions ---
            print("4. Parsing and executing actions...")
            actions = result.get("actions", [])
            if not actions:
                print("No actions received from the model. Skipping.")
                continue

            # Execute full chunk (0..chunk_size-1). Set SKIP_FIRST_K to skip warm-up steps if needed.
            SKIP_FIRST_K = 0
            for i, act in enumerate(actions):
                if i < SKIP_FIRST_K:
                    continue
                action = np.array(act, dtype=np.float32)
                if action.shape[0] != 7:
                    print(f"Warning: Action dimension is {action.shape[0]}, expected 7. Skipping.")
                    continue
                
                target_pose = action[:6]
                target_gripper = action[6]
                print(f"Target Pose: {np.round(target_pose, 3)}, Gripper: {target_gripper:.2f}")
                print(f"[Step {i+1}/{len(actions)}] Executing action...")
                control_franka_pose(target_pose.tolist(), target_gripper.item())
                time.sleep(0.05) # Short pause between actions

            print("Action sequence execution complete.")
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # --- C. SHUTDOWN ---
        if cameras:
            print("Stopping cameras.")
            cameras.stop()
        print("Shutdown complete. Exiting program.")

if __name__ == '__main__':
    main()