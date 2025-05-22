# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test script for VR teleoperation functionality.
This script simulates an Oculus Quest VR controller sending data to the robot.
"""

import argparse
import numpy as np
import time
import zmq
import threading
import logging
from scipy.spatial.transform import Rotation

# Constants for VR teleoperation
ARM_TELEOP_STOP = 0
ARM_TELEOP_CONT = 1
ARM_HIGH_RESOLUTION = 0
ARM_LOW_RESOLUTION = 1

# Define the Oculus joints mapping
OCULUS_JOINTS = {
    'wrist': [0],
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20]
}


class ZMQKeypointPusher:
    """Pusher for ZMQ keypoint data using PUSH socket."""
    def __init__(self, host, port, max_retries=5, heartbeat_interval=0.5):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        
        # Try to connect with retries and port increment
        connected = False
        retry_count = 0
        self.port = port
        self.host = host
        
        while not connected and retry_count < max_retries:
            try:
                connection_string = f"tcp://{host}:{self.port}"
                self.socket.connect(connection_string)
                connected = True
                print(f"Successfully connected to {connection_string}")
            except zmq.error.ZMQError as e:
                retry_count += 1
                print(f"Failed to connect to {host}:{self.port}: {e}")
                if retry_count < max_retries:
                    self.port += 1000  # Try a port 1000 higher
                    print(f"Retrying with port {self.port}")
                    time.sleep(1)  # Wait a bit before retrying
                else:
                    raise Exception(f"Failed to connect after {max_retries} retries")
        
        # Setup heartbeat
        self.heartbeat_interval = heartbeat_interval
        self.last_heartbeat_time = 0
        
    def push_keypoints(self, keypoints, message_type="keypoints"):
        """Push keypoints with a message type identifier and timestamp."""
        message = {
            'type': message_type,
            'data': keypoints,
            'timestamp': time.time()
        }
        self.socket.send_pyobj(message)
    
    def send_heartbeat(self):
        """Send a heartbeat message to keep the connection alive."""
        current_time = time.time()
        if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
            self.push_keypoints(np.array([current_time]), "heartbeat")
            self.last_heartbeat_time = current_time
        
    def get_port(self):
        return self.port


def generate_hand_frame(t):
    """
    Generate a simulated hand frame for testing.
    
    Args:
        t (float): Time parameter to create movement
        
    Returns:
        np.ndarray: 4x3 hand frame matrix
    """
    # Create a translation that moves in a circle
    x = 0.1 * np.sin(t)
    y = 0.1 * np.cos(t)
    z = 0.5 + 0.05 * np.sin(t * 0.5)
    
    # Create a rotation that changes over time
    rotation = Rotation.from_euler('xyz', [t * 0.2, t * 0.1, t * 0.15]).as_matrix()
    
    # Create the hand frame
    hand_frame = np.zeros((4, 3))
    hand_frame[0] = [x, y, z]  # Translation
    hand_frame[1:] = rotation  # Rotation
    
    return hand_frame


def generate_hand_keypoints(t):
    """
    Generate simulated hand keypoints for testing.
    
    Args:
        t (float): Time parameter to create movement
        
    Returns:
        np.ndarray: Array of hand keypoints
    """
    # Create a base hand pose
    keypoints = np.zeros((21, 3))
    
    # Wrist position
    keypoints[0] = [0, 0, 0]
    
    # Thumb
    keypoints[1] = [0.03, 0.01, 0]
    keypoints[2] = [0.05, 0.02, 0]
    keypoints[3] = [0.07, 0.03, 0]
    keypoints[4] = [0.09, 0.04, 0]
    
    # Index finger
    keypoints[5] = [0.02, 0.04, 0]
    keypoints[6] = [0.03, 0.08, 0]
    keypoints[7] = [0.04, 0.12, 0]
    keypoints[8] = [0.05, 0.16, 0]
    
    # Middle finger
    keypoints[9] = [0, 0.05, 0]
    keypoints[10] = [0, 0.09, 0]
    keypoints[11] = [0, 0.13, 0]
    keypoints[12] = [0, 0.17, 0]
    
    # Ring finger
    keypoints[13] = [-0.02, 0.04, 0]
    keypoints[14] = [-0.03, 0.08, 0]
    keypoints[15] = [-0.04, 0.12, 0]
    keypoints[16] = [-0.05, 0.16, 0]
    
    # Pinky finger
    keypoints[17] = [-0.04, 0.03, 0]
    keypoints[18] = [-0.06, 0.06, 0]
    keypoints[19] = [-0.08, 0.09, 0]
    keypoints[20] = [-0.10, 0.12, 0]
    
    # Add some movement based on time
    # Simulate pinching motion between thumb and pinky for gripper control
    if int(t) % 4 >= 2:  # Every 2 seconds, alternate between pinch and open
        # Pinch position
        keypoints[4] = [-0.08, 0.08, 0]  # Move thumb tip closer to pinky
        keypoints[20] = [-0.08, 0.08, 0]  # Move pinky tip closer to thumb
    
    # Simulate pinching motion between thumb and middle/ring for pause control
    if int(t) % 6 >= 3:  # Every 3 seconds, alternate between pinch and open
        # Pinch position
        keypoints[4] = [0, 0.13, 0]  # Move thumb tip closer to middle finger
        keypoints[12] = [0.04, 0.13, 0]  # Move middle finger tip closer to thumb
    
    return keypoints


def run_vr_simulation(host="localhost", duration=30, fps=30):
    """
    Run a VR simulation that publishes hand frames and keypoints.
    
    Args:
        host (str): Host address
        duration (int): Duration of the simulation in seconds
        fps (int): Frames per second
    """
    try:
        # Create publishers with automatic port selection if default is in use
        print(f"Initializing publishers on {host}...")
        
        # We'll use a single pusher for all message types
        keypoint_port = 8087  # Start with this port - correct port for the VR app
        keypoint_pusher = ZMQKeypointPusher(host, keypoint_port)
        actual_keypoint_port = keypoint_pusher.port
        
        print(f"Starting VR simulation on {host}")
        print(f"Pushing data to port {actual_keypoint_port}")
        print(f"Running for {duration} seconds at {fps} FPS")
        print(f"NOTE: If you're running the VR teleoperation, make sure to update the port in your command:")
        print(f"  --control.transformed_keypoint_port={actual_keypoint_port}")
    
        # Run the simulation
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                t = time.time() - start_time
                
                # Generate and push hand frame
                hand_frame = generate_hand_frame(t)
                keypoint_pusher.push_keypoints(hand_frame, "transformed_hand_frame")
                
                # Generate and push hand keypoints
                hand_keypoints = generate_hand_keypoints(t)
                keypoint_pusher.push_keypoints(hand_keypoints, "transformed_hand_coords")
                
                # Push resolution button state (alternate between high and low resolution)
                resolution = ARM_HIGH_RESOLUTION if int(t) % 8 >= 4 else ARM_LOW_RESOLUTION
                keypoint_pusher.push_keypoints(np.array([resolution]), "button")
                
                # Send heartbeat if needed
                keypoint_pusher.send_heartbeat()
                
                # Sleep to maintain FPS
                frame_count += 1
                elapsed = time.time() - start_time
                target_elapsed = frame_count / fps
                if elapsed < target_elapsed:
                    time.sleep(target_elapsed - elapsed)
                
                # Print status every second
                if int(elapsed) > int(elapsed - 0.1):
                    print(f"Simulation running for {int(elapsed)}s, publishing at {frame_count / elapsed:.1f} FPS")
        
        except KeyboardInterrupt:
            print("Simulation interrupted by user")
            # Send one final message to indicate clean shutdown
            keypoint_pusher.push_keypoints(np.array([0]), "shutdown")
            time.sleep(0.5)  # Give time for the message to be sent
        
        print(f"Simulation completed after {time.time() - start_time:.1f} seconds")
        print(f"Published {frame_count} frames at an average of {frame_count / (time.time() - start_time):.1f} FPS")
    
    except Exception as e:
        print(f"Error in VR simulation: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VR Teleoperation Test")
    parser.add_argument("--host", type=str, default="192.168.0.117", help="Host address")
    parser.add_argument("--duration", type=int, default=1, help="Duration of the simulation in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    
    args = parser.parse_args()
    run_vr_simulation(args.host, args.duration, args.fps)