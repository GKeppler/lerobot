#!/usr/bin/env python3
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
Verification script for VR teleoperation.
This script connects to the ZMQ publishers created by test_vr_teleoperation.py
and verifies that it can receive the data correctly.
"""

import argparse
import numpy as np
import time
import zmq
import logging
from scipy.spatial.transform import Rotation

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


class ZMQKeypointSubscriber:
    """Subscriber for ZMQ keypoint data."""
    def __init__(self, host, port, topic=None):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        connection_string = f"tcp://{host}:{port}"
        logging.info(f"Connecting to {connection_string}")
        self.socket.connect(connection_string)
        if topic is None:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            logging.info(f"Subscribing to all topics")
        else:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
            logging.info(f"Subscribing to topic '{topic}'")
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

    def recv_keypoints(self, flags=0, timeout=1000):
        """
        Receive keypoints from the ZMQ socket.
        
        Args:
            flags (int): ZMQ flags
            timeout (int): Timeout in milliseconds
            
        Returns:
            tuple: (topic, data) or (None, None) if no data is available
        """
        if flags == zmq.NOBLOCK:
            socks = dict(self.poller.poll(timeout))
            if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                try:
                    topic = self.socket.recv_string(flags=flags)
                    data = self.socket.recv_pyobj(flags=flags)
                    return topic, data
                except zmq.ZMQError:
                    return None, None
            return None, None
        else:
            socks = dict(self.poller.poll(timeout))
            if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                topic = self.socket.recv_string()
                data = self.socket.recv_pyobj()
                return topic, data
            return None, None


def verify_vr_connection(host="localhost", hand_frame_port=10089, resolution_button_port=8093, duration=30):
    """
    Verify that we can connect to the ZMQ publishers and receive data.
    
    Args:
        host (str): Host address
        hand_frame_port (int): Port for hand frame and keypoints
        resolution_button_port (int): Port for resolution button
        duration (int): Duration of the verification in seconds
    """
    logging.info(f"Verifying VR connection to {host}")
    logging.info(f"Hand frame port: {hand_frame_port}")
    logging.info(f"Resolution button port: {resolution_button_port}")
    
    # Create subscribers
    hand_frame_subscriber = ZMQKeypointSubscriber(host, hand_frame_port)
    resolution_button_subscriber = ZMQKeypointSubscriber(host, resolution_button_port, topic="button")
    
    # Run the verification
    start_time = time.time()
    received_hand_frame = False
    received_hand_keypoints = False
    received_resolution_button = False
    
    while time.time() - start_time < duration:
        # Check for hand frame
        topic, data = hand_frame_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
        if topic is not None:
            if topic == "transformed_hand_frame":
                if not received_hand_frame:
                    logging.info(f"Received hand frame: {data.shape}")
                    received_hand_frame = True
            elif topic == "transformed_hand_coords":
                if not received_hand_keypoints:
                    logging.info(f"Received hand keypoints: {data.shape}")
                    received_hand_keypoints = True
        
        # Check for resolution button
        topic, data = resolution_button_subscriber.recv_keypoints(flags=zmq.NOBLOCK)
        if topic is not None and topic == "button":
            if not received_resolution_button:
                logging.info(f"Received resolution button: {data}")
                received_resolution_button = True
        
        # If we've received all types of data, we're done
        if received_hand_frame and received_hand_keypoints and received_resolution_button:
            logging.info("Successfully received all types of data!")
            break
        
        # Sleep a bit to avoid busy waiting
        time.sleep(0.1)
    
    # Check if we received all types of data
    if not received_hand_frame:
        logging.error("Did not receive any hand frames")
    if not received_hand_keypoints:
        logging.error("Did not receive any hand keypoints")
    if not received_resolution_button:
        logging.error("Did not receive any resolution button data")
    
    if received_hand_frame and received_hand_keypoints and received_resolution_button:
        logging.info("Verification successful!")
        return True
    else:
        logging.error("Verification failed!")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VR Teleoperation Verification")
    parser.add_argument("--host", type=str, default="localhost", help="Host address")
    parser.add_argument("--hand-frame-port", type=int, default=8087, help="Port for hand frame and keypoints")
    parser.add_argument("--resolution-button-port", type=int, default=8093, help="Port for resolution button")
    parser.add_argument("--duration", type=int, default=10, help="Duration of the verification in seconds")
    
    args = parser.parse_args()
    verify_vr_connection(args.host, args.hand_frame_port, args.resolution_button_port, args.duration)