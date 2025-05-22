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
# import threading # Not used
# import logging # Not used directly, print is used
from scipy.spatial.transform import Rotation

# Constants for VR teleoperation
ARM_TELEOP_STOP = 0
ARM_TELEOP_CONT = 1
ARM_HIGH_RESOLUTION = 0
ARM_LOW_RESOLUTION = 1

# Define the Oculus joints mapping (copied for completeness, not strictly needed by pusher)
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
        # Set a High Water Mark on the sender too. If the receiver is slow,
        # this will cause send to block or error rather than buffer indefinitely.
        self.socket.set_hwm(10) 
        
        connected = False
        retry_count = 0
        self.port = port
        self.host = host
        
        while not connected and retry_count < max_retries:
            try:
                connection_string = f"tcp://{host}:{self.port}"
                print(f"Attempting to connect PUSH socket to {connection_string}...")
                self.socket.connect(connection_string)
                connected = True
                print(f"Successfully connected PUSH socket to {connection_string}")
            except zmq.error.ZMQError as e:
                retry_count += 1
                print(f"Failed to connect to {host}:{self.port}: {e}")
                if retry_count < max_retries:
                    # Port increment on client side is not standard if server binds to a fixed (or auto-selected then fixed) port.
                    # Client should retry connection to the *same* port the server is on.
                    # self.port += 1000 # Removing this, client should know server port.
                    print(f"Retrying connection to {host}:{self.port}...")
                    time.sleep(1)
                else:
                    print(f"Failed to connect PUSH socket after {max_retries} retries.")
                    raise # Re-raise the exception to stop the script
        
        self.heartbeat_interval = heartbeat_interval
        self.last_heartbeat_time = 0
        
    def push_keypoints(self, keypoints, message_type="keypoints"):
        message = {
            'type': message_type,
            'data': keypoints,
            'timestamp': time.time()
        }
        try:
            self.socket.send_pyobj(message, flags=zmq.NOBLOCK) # Try non-blocking send
        except zmq.Again:
            print(f"Warning: ZMQ PUSH socket buffer full (sending {message_type}). Message might be dropped or delayed.")
            # Optionally, implement retry or blocking send here if NOBLOCK is too aggressive
            # self.socket.send_pyobj(message) # Fallback to blocking send

    def send_heartbeat(self):
        current_time = time.time()
        if current_time - self.last_heartbeat_time >= self.heartbeat_interval:
            self.push_keypoints(np.array([current_time]), "heartbeat")
            self.last_heartbeat_time = current_time
        
    def get_port(self): # Port it's trying to connect to
        return self.port

    def close(self):
        print(f"Closing ZMQ PUSH socket connected to {self.host}:{self.port}")
        if hasattr(self, 'socket') and self.socket and not self.socket.closed:
            self.socket.close(linger=0)
        if hasattr(self, 'context') and self.context and not self.context.closed:
            self.context.term()


def generate_hand_frame(t):
    # Circular motion in XY plane, sinusoidal in Z
    x = 0.1 * np.sin(t * 0.5)  # Slower XY circle
    y = 0.1 * np.cos(t * 0.5)
    z = 0.05 * np.sin(t)     # Base Z at 0, moves +/- 0.05m (robot relative)
    
    # Rotation changing over time
    # Make rotations smoother and less extreme
    rotation = Rotation.from_euler('xyz', [0.5 * np.sin(t * 0.2), 
                                           0.5 * np.cos(t * 0.1), 
                                           0.3 * np.sin(t * 0.15)]).as_matrix()
    
    hand_frame = np.zeros((4, 3))
    hand_frame[0] = [x, y, z]  # Translation (relative to initial robot EE or a world frame)
    hand_frame[1:] = rotation  # Rotation matrix R (orientation of hand in world/robot base)
                               # Columns of R are X,Y,Z axes of hand frame, expressed in world frame.
    return hand_frame


def generate_hand_keypoints(t, cycle_duration=6):
    keypoints = np.zeros((21, 3)) # Base keypoints around origin
    # ... (keypoint definitions from original script are fine) ...
    # Wrist position (origin for other keypoints)
    keypoints[0] = [0, 0, 0]
    # Thumb
    keypoints[OCULUS_JOINTS['thumb'][0]] = [0.01, 0.005, 0]
    keypoints[OCULUS_JOINTS['thumb'][1]] = [0.02, 0.01, 0]
    keypoints[OCULUS_JOINTS['thumb'][2]] = [0.03, 0.015, 0]
    keypoints[OCULUS_JOINTS['thumb'][3]] = [0.04, 0.02, 0] # Thumb tip
    # Index
    keypoints[OCULUS_JOINTS['index'][0]] = [0.005, 0.025, 0] 
    # ... (fill in other keypoints if their precise location matters for gestures) ...
    keypoints[OCULUS_JOINTS['index'][3]] = [0.02, 0.08, 0] # Index tip
    # Middle
    keypoints[OCULUS_JOINTS['middle'][0]] = [0, 0.03, 0]
    keypoints[OCULUS_JOINTS['middle'][3]] = [0, 0.09, 0] # Middle tip
    # Ring
    keypoints[OCULUS_JOINTS['ring'][0]] = [-0.005, 0.025, 0]
    keypoints[OCULUS_JOINTS['ring'][3]] = [-0.02, 0.08, 0] # Ring tip
    # Pinky
    keypoints[OCULUS_JOINTS['pinky'][0]] = [-0.01, 0.02, 0]
    keypoints[OCULUS_JOINTS['pinky'][3]] = [-0.04, 0.06, 0] # Pinky tip

    # Simulate pinching for gripper: Thumb tip to Pinky tip (distance < 0.03)
    # Cycle: 0-2s open, 2-4s pinch, 4-6s open
    time_in_gripper_cycle = t % (cycle_duration / 1.5) # 4s cycle for gripper
    if time_in_gripper_cycle >= (cycle_duration / 3): # Pinch for 2s out of 4s
        # Move thumb tip and pinky tip close together
        keypoints[OCULUS_JOINTS['thumb'][3]] = [-0.01, 0.03, 0.005] 
        keypoints[OCULUS_JOINTS['pinky'][3]] = [-0.01, 0.035, 0.005]
    
    # Simulate pinching for pause: Thumb tip to Middle/Ring tip
    # Cycle: 0-3s no pause pinch, 3-6s pause pinch
    time_in_pause_cycle = t % cycle_duration # 6s cycle for pause
    if time_in_pause_cycle >= (cycle_duration / 2): # Pinch for 3s out of 6s
        # Move thumb tip and middle tip close
        keypoints[OCULUS_JOINTS['thumb'][3]] = [0, 0.05, 0.005]
        keypoints[OCULUS_JOINTS['middle'][3]] = [0.005, 0.055, 0.005]
    
    return keypoints


def run_vr_simulation(host="localhost", keypoint_port=8087, duration=30, fps=30):
    pusher = None
    try:
        print(f"Initializing ZMQKeypointPusher to connect to {host}:{keypoint_port}...")
        pusher = ZMQKeypointPusher(host, keypoint_port)
        
        print(f"Starting VR simulation pushing to {host}:{pusher.get_port()}") # get_port() returns configured port
        print(f"Running for {duration} seconds at approximately {fps} FPS.")
        print(f"Target server port for PULL socket is {keypoint_port}.")
        print("Ensure the lerobot script's --control.transformed_keypoint_port matches this.")
    
        start_time = time.time()
        frame_count = 0
        
        sim_active = True
        while sim_active and (time.time() - start_time < duration):
            loop_start_time = time.time()
            t_sim = loop_start_time - start_time # Simulation time
            
            # Generate and push hand frame
            hand_frame = generate_hand_frame(t_sim)
            pusher.push_keypoints(hand_frame, "transformed_hand_frame")
            
            # Generate and push hand keypoints
            hand_keypoints = generate_hand_keypoints(t_sim)
            pusher.push_keypoints(hand_keypoints, "transformed_hand_coords")
            
            # Push resolution button state (alternate every 4s within an 8s cycle)
            resolution = ARM_HIGH_RESOLUTION if (int(t_sim) % 8) < 4 else ARM_LOW_RESOLUTION
            pusher.push_keypoints(np.array([resolution]), "button")
            
            pusher.send_heartbeat()
            
            frame_count += 1
            
            # Precise sleep to maintain FPS
            elapsed_in_loop = time.time() - loop_start_time
            sleep_time = (1.0 / fps) - elapsed_in_loop
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            if frame_count % fps == 0: # Print status roughly every second
                print(f"Sim running for {int(t_sim)}s. Frame {frame_count}. Actual FPS: {frame_count / t_sim if t_sim > 0 else fps:.1f}")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"Error in VR simulation: {e}", exc_info=True)
    finally:
        if pusher:
            print("Sending shutdown signal...")
            pusher.push_keypoints(np.array([0]), "shutdown") # Send shutdown signal
            time.sleep(0.5) # Give time for message to be sent
            pusher.close()
        
        total_time = time.time() - start_time
        print(f"Simulation ended after {total_time:.1f} seconds.")
        if total_time > 0 :
             print(f"Published {frame_count} frames at an average of {frame_count / total_time:.1f} FPS.")
        else:
            print(f"Published {frame_count} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VR Teleoperation Test Script")
    parser.add_argument("--host", type=str, default="192.168.0.117", help="Host address of the robot control script")
    parser.add_argument("--port", type=int, default=8087, help="Port number for ZMQ PUSH socket (must match robot script's PULL port)")
    parser.add_argument("--duration", type=int, default=60, help="Duration of the simulation in seconds") # Increased default
    parser.add_argument("--fps", type=int, default=30, help="Target frames per second for publishing data")
    
    args = parser.parse_args()
    run_vr_simulation(args.host, args.port, args.duration, args.fps)