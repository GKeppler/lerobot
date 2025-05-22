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

import numpy as np
import time
import zmq
import threading
import logging
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import rerun as rr
from scipy.spatial.transform import Rotation, Slerp
from numpy.linalg import pinv

from lerobot.common.robot_devices.control_configs import VRTeleoperateControlConfig
from lerobot.common.robot_devices.control_utils import control_loop
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.utils.utils import log_say


# Constants for VR teleoperation
ARM_TELEOP_STOP = 0
ARM_TELEOP_CONT = 1
ARM_HIGH_RESOLUTION = 0
ARM_LOW_RESOLUTION = 1
SCALE_FACTOR = 1000.0  # Convert mm to m
BIMANUAL_VR_FREQ = 90  # Hz

# Define the Oculus joints mapping
OCULUS_JOINTS = {
    'wrist': [0],
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20]
}


# Filter for removing noise in the teleoperation
class Filter:
    def __init__(self, state, comp_ratio=0.6):
        self.pos_state = state[:3]
        self.ori_state = state[3:7]
        self.comp_ratio = comp_ratio

    def __call__(self, next_state):
        self.pos_state = self.pos_state[:3] * self.comp_ratio + next_state[:3] * (1 - self.comp_ratio)
        ori_interp = Slerp([0, 1], Rotation.from_rotvec(
            np.stack([self.ori_state, next_state[3:6]], axis=0)),)
        self.ori_state = ori_interp([1 - self.comp_ratio])[0].as_rotvec()
        return np.concatenate([self.pos_state, self.ori_state])


class ZMQKeypointPuller:
    """Puller for ZMQ keypoint data using PULL socket."""
    def __init__(self, port, max_retries=3, data_timeout=1.0):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        
        # Try to bind with retries
        bound = False
        retry_count = 0
        self.port = port
        
        while not bound and retry_count < max_retries:
            try:
                # Bind to all interfaces
                bind_address = f"tcp://0.0.0.0:{self.port}"
                logging.info(f"Attempting to bind to {bind_address}")
                self.socket.bind(bind_address)
                bound = True
                logging.info(f"Successfully bound to {bind_address}")
            except zmq.error.ZMQError as e:
                retry_count += 1
                logging.warning(f"Failed to bind to port {self.port}: {e}")
                if retry_count < max_retries:
                    self.port += 1000  # Try a port 1000 higher
                    logging.info(f"Retrying with port {self.port}")
                    time.sleep(1)  # Wait a bit before retrying
                else:
                    logging.error(f"Failed to bind after {max_retries} retries")
                    raise
        
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        
        # Store message types, their latest data, and timestamps
        self.message_data = {}
        self.message_timestamps = {}
        self.last_receive_time = time.time()
        self.data_timeout = data_timeout  # Timeout in seconds

    def recv_keypoints(self, flags=0):
        if flags == zmq.NOBLOCK:
            socks = dict(self.poller.poll(0))
            if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                try:
                    message = self.socket.recv_pyobj(flags=flags)
                    self.last_receive_time = time.time()
                    if isinstance(message, dict) and 'type' in message and 'data' in message:
                        message_type = message['type']
                        self.message_data[message_type] = message['data']
                        self.message_timestamps[message_type] = time.time()
                        return message['data']
                    return message
                except zmq.ZMQError:
                    return None
            return None
        else:
            message = self.socket.recv_pyobj()
            self.last_receive_time = time.time()
            if isinstance(message, dict) and 'type' in message and 'data' in message:
                message_type = message['type']
                self.message_data[message_type] = message['data']
                self.message_timestamps[message_type] = time.time()
                return message['data']
            return message
    
    def get_latest_data(self, message_type):
        """Get the latest data for a specific message type."""
        return self.message_data.get(message_type)
    
    def is_data_fresh(self, message_type=None):
        """
        Check if data is fresh (received within timeout period).
        
        Args:
            message_type: Specific message type to check, or None to check any message
            
        Returns:
            bool: True if data is fresh, False otherwise
        """
        current_time = time.time()
        
        # If no specific message type, check if any data has been received recently
        if message_type is None:
            return (current_time - self.last_receive_time) < self.data_timeout
        
        # Check if the specific message type exists and is fresh
        if message_type in self.message_timestamps:
            return (current_time - self.message_timestamps[message_type]) < self.data_timeout
        
        return False


# class ZMQKeypointPublisher:
#     """Publisher for ZMQ keypoint data."""
#     def __init__(self, host, port):
#         self.context = zmq.Context()
#         self.socket = self.context.socket(zmq.PUB)
#         # Always bind to all interfaces (0.0.0.0) to allow connections from any IP
#         bind_host = "0.0.0.0"
#         self.socket.bind(f"tcp://{bind_host}:{port}")

#     def pub_keypoints(self, keypoints, topic="keypoints"):
#         self.socket.send_string(topic, zmq.SNDMORE)
#         self.socket.send_pyobj(keypoints)


class VRTeleoperator:
    """Class for VR teleoperation of a robot."""
    def __init__(self, robot: Robot, cfg: VRTeleoperateControlConfig):
        self.robot = robot
        self.cfg = cfg
        
        # Initialize ZMQ subscribers and publishers
        # Use localhost for subscribers when running locally
        # This ensures that the subscribers can connect to the publishers
        # even when the publisher is bound to 0.0.0.0
        subscriber_host = "localhost" if cfg.host_ip == "localhost" else cfg.host_ip
        
        logging.info(f"Initializing ZMQ subscribers with host: {subscriber_host}")
        
        # Initialize a single PULL socket for all message types
        self._keypoint_puller = ZMQKeypointPuller(
            port=cfg.transformed_keypoint_port
        )
        
        # Store the port for logging
        self.keypoint_port = self._keypoint_puller.port
        logging.info(f"VR Teleoperation PULL socket bound to port {self.keypoint_port}")
        
        # Initialize state variables
        self.gripper_flag = 1
        self.pause_flag = 1
        self.prev_pause_flag = 0
        self.is_first_frame = True
        self.gripper_cnt = 0
        self.prev_gripper_flag = 0
        self.pause_cnt = 0
        self.gripper_correct_state = 1
        self.resolution_scale = 1
        self.arm_teleop_state = ARM_TELEOP_STOP
        
        # We'll initialize joint positions when the robot is connected
        self.initial_joint_positions = {}
        self.joint_filters = {}
        self.use_filter = cfg.use_filter
        self.initialized = False
        
        # Save the original teleop_step method
        self.original_teleop_step = robot.teleop_step
        # Replace with our VR teleop_step method
        robot.teleop_step = self.vr_teleop_step
        
        logging.info("VR Teleoperator initialized")

    def robot_pose_aa_to_affine(self, pose_aa: np.ndarray) -> np.ndarray:
        """
        Converts a robot pose in axis-angle format to an affine matrix.
        
        Args:
            pose_aa (np.ndarray): [x, y, z, ax, ay, az] where (x, y, z) is the position 
                                  and (ax, ay, az) is the axis-angle rotation.
        
        Returns:
            np.ndarray: 4x4 affine matrix [[R, t],[0, 1]]
        """
        rotation = Rotation.from_rotvec(pose_aa[3:]).as_matrix()
        translation = np.array(pose_aa[:3])

        return np.block([[rotation, translation[:, np.newaxis]],[0, 0, 0, 1]])

    def _get_hand_frame(self):
        """
        Get the transformed hand frame.
        
        Returns:
            np.ndarray: Hand frame as a 4x3 matrix
        """
        # Try to receive new data
        for i in range(10):
            data = self._keypoint_puller.recv_keypoints(flags=zmq.NOBLOCK)
            if data is not None:
                break
        
        # Get the latest hand frame data
        data = self._keypoint_puller.get_latest_data('transformed_hand_frame')
        if data is None:
            return None
        return np.asanyarray(data).reshape(4, 3)
    
    def _get_resolution_scale_mode(self):
        """
        Get the resolution scale mode.
        
        Returns:
            int: Resolution scale mode
        """
        # Try to receive new data
        self._keypoint_puller.recv_keypoints(flags=zmq.NOBLOCK)
        
        # Get the latest resolution data
        data = self._keypoint_puller.get_latest_data('button')
        if data is None:
            # Default to high resolution if no data is available
            return ARM_HIGH_RESOLUTION
        
        res_scale = np.asanyarray(data).reshape(1)[0]  # Make sure this data is one dimensional
        return res_scale
    
    def _get_arm_teleop_state_from_hand_keypoints(self):
        """
        Get the arm teleoperation state from hand keypoints.
        
        Returns:
            tuple: (pause_state, pause_status, pause_right)
        """
        pause_state, pause_status, pause_right = self.get_pause_state_from_hand_keypoints()
        pause_status = np.asanyarray(pause_status).reshape(1)[0] 

        return pause_state, pause_status, pause_right

    def _turn_frame_to_homo_mat(self, frame):
        """
        Turn a frame to a homogeneous matrix.
        
        Args:
            frame (np.ndarray): Frame as a 4x3 matrix
        
        Returns:
            np.ndarray: 4x4 homogeneous matrix
        """
        t = frame[0]
        R = frame[1:]

        homo_mat = np.zeros((4, 4))
        homo_mat[:3, :3] = np.transpose(R)
        homo_mat[:3, 3] = t
        homo_mat[3, 3] = 1

        return homo_mat

    def _homo2cart(self, homo_mat):
        """
        Turn homogeneous matrix to cartesian vector.
        
        Args:
            homo_mat (np.ndarray): 4x4 homogeneous matrix
        
        Returns:
            np.ndarray: Cartesian vector [x, y, z, rx, ry, rz]
        """
        t = homo_mat[:3, 3]
        R = Rotation.from_matrix(
            homo_mat[:3, :3]).as_rotvec(degrees=False)

        cart = np.concatenate(
            [t, R], axis=0
        )
        return cart
    
    def _get_scaled_cart_pose(self, moving_robot_homo_mat):
        """
        Get the scaled cartesian pose.
        
        Args:
            moving_robot_homo_mat (np.ndarray): 4x4 homogeneous matrix of the moving robot
        
        Returns:
            np.ndarray: Scaled cartesian pose [x, y, z, rx, ry, rz]
        """
        # Get the cart pose without the scaling
        unscaled_cart_pose = self._homo2cart(moving_robot_homo_mat)

        # Get the current cart pose
        home_pose = self.robot.get_cartesian_pose()
        home_pose_array = np.array(home_pose)
        current_cart_pose = home_pose_array

        # Get the difference in translation between these two cart poses
        diff_in_translation = unscaled_cart_pose[:3] - current_cart_pose[:3]
        scaled_diff_in_translation = diff_in_translation * self.resolution_scale
        
        scaled_cart_pose = np.zeros(6)
        scaled_cart_pose[3:] = unscaled_cart_pose[3:]  # Get the rotation directly
        scaled_cart_pose[:3] = current_cart_pose[:3] + scaled_diff_in_translation  # Get the scaled translation only

        return scaled_cart_pose

    def _initialize(self):
        """
        Initialize the VR teleoperator with the current robot state.
        This should be called after the robot is connected.
        """
        if not self.robot.is_connected:
            logging.warning("Robot is not connected. Cannot initialize VR teleoperator.")
            return False
            
        try:
            # Initialize joint positions
            self.initial_joint_positions = {}
            for name in self.robot.follower_arms:
                joint_pos = self.robot.follower_arms[name].read("Present_Position")
                self.initial_joint_positions[name] = torch.from_numpy(joint_pos)
            
            # Initialize filters
            if self.use_filter:
                self.joint_filters = {}
                for name, pos in self.initial_joint_positions.items():
                    self.joint_filters[name] = Filter(pos.numpy(), comp_ratio=self.cfg.filter_ratio)
            
            self.initialized = True
            logging.info("VR teleoperator initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Error initializing VR teleoperator: {e}")
            return False

    def _reset_teleop(self):
        """
        Reset teleoperation and make the current frame as initial frame.
        
        Returns:
            None
        """
        logging.info('****** RESETTING TELEOP ****** ')
        
        # Make sure we're initialized
        if not self.initialized:
            success = self._initialize()
            if not success:
                logging.error("Failed to initialize VR teleoperator during reset")
                return
        
        # Reset joint positions
        self.initial_joint_positions = {}
        for name in self.robot.follower_arms:
            joint_pos = self.robot.follower_arms[name].read("Present_Position")
            self.initial_joint_positions[name] = torch.from_numpy(joint_pos)
        
        # Reset filters
        if self.use_filter:
            self.joint_filters = {}
            for name, pos in self.initial_joint_positions.items():
                self.joint_filters[name] = Filter(pos.numpy(), comp_ratio=self.cfg.filter_ratio)
        
        # Get initial hand frame
        first_hand_frame = self._get_hand_frame()
        while first_hand_frame is None:
            first_hand_frame = self._get_hand_frame()
        
        self.hand_init_frame = first_hand_frame
        self.is_first_frame = False
        logging.info("Resetting complete")

    def get_gripper_state_from_hand_keypoints(self):
        """
        Get gripper state from hand keypoints.
        
        Returns:
            tuple: (gripper_state, status, gripper_fl)
        """
        # Try to receive new data
        self._keypoint_puller.recv_keypoints(flags=zmq.NOBLOCK)
        
        # Get the latest hand coordinates
        transformed_hand_coords = self._keypoint_puller.get_latest_data('transformed_hand_coords')
        if transformed_hand_coords is None:
            return self.gripper_flag, False, False
        distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['pinky'][-1]] - transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
        thresh = 0.03
        gripper_fl = False
        if distance < thresh:
            self.gripper_cnt += 1
            if self.gripper_cnt == 1:
                self.prev_gripper_flag = self.gripper_flag
                self.gripper_flag = not self.gripper_flag 
                gripper_fl = True
        else: 
            self.gripper_cnt = 0
        gripper_state = np.asanyarray(self.gripper_flag).reshape(1)[0]
        status = False  
        if gripper_state != self.prev_gripper_flag:
            status = True
        return gripper_state, status, gripper_fl 
   
    def get_pause_state_from_hand_keypoints(self):
        """
        Toggle the robot to pause/resume using ring/middle finger pinch.
        
        Returns:
            tuple: (pause_state, pause_status, pause_right)
        """
        # Try to receive new data
        self._keypoint_puller.recv_keypoints(flags=zmq.NOBLOCK)
        
        # Get the latest hand coordinates
        transformed_hand_coords = self._keypoint_puller.get_latest_data('transformed_hand_coords')
        if transformed_hand_coords is None:
            return self.pause_flag, False, True
        ring_distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['ring'][-1]] - transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
        middle_distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['middle'][-1]] - transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
        thresh = 0.03 
        pause_right = True
        if ring_distance < thresh or middle_distance < thresh:
            self.pause_cnt += 1
            if self.pause_cnt == 1:
                self.prev_pause_flag = self.pause_flag
                self.pause_flag = not self.pause_flag       
        else:
            self.pause_cnt = 0
        pause_state = np.asanyarray(self.pause_flag).reshape(1)[0]
        pause_status = False  
        if pause_state != self.prev_pause_flag:
            pause_status = True 
        return pause_state, pause_status, pause_right

    def apply_retargeted_angles(self):
        """
        Apply retargeted angles to the robot.
        
        Returns:
            dict: Action to send to the robot
        """
        # Make sure we're initialized
        if not self.initialized:
            success = self._initialize()
            if not success:
                logging.error("Failed to initialize VR teleoperator during apply_retargeted_angles")
                return None
        
        # See if there is a reset in the teleop state
        new_arm_teleop_state, pause_status, pause_right = self._get_arm_teleop_state_from_hand_keypoints()
        if self.is_first_frame or (self.arm_teleop_state == ARM_TELEOP_STOP and new_arm_teleop_state == ARM_TELEOP_CONT):
            self._reset_teleop()  # Reset the teleop state
        
        self.arm_teleop_state = new_arm_teleop_state
        arm_teleoperation_scale_mode = self._get_resolution_scale_mode()

        if arm_teleoperation_scale_mode == ARM_HIGH_RESOLUTION:
            self.resolution_scale = 1
        elif arm_teleoperation_scale_mode == ARM_LOW_RESOLUTION:
            self.resolution_scale = 0.6

        # Get hand frame
        moving_hand_frame = self._get_hand_frame()
        if moving_hand_frame is None:
            return None  # It means we are not on the arm mode yet

        # Process hand frame to get joint positions
        # For simplicity, we'll map hand movements to joint position changes
        # This is a simplified approach - in a real implementation, you would use inverse kinematics
        
        # Get current joint positions
        current_joint_positions = {}
        for name in self.robot.follower_arms:
            joint_pos = self.robot.follower_arms[name].read("Present_Position")
            current_joint_positions[name] = torch.from_numpy(joint_pos)
        
        # Calculate joint position changes based on hand movement
        joint_position_changes = {}
        for name, initial_pos in self.initial_joint_positions.items():
            # Extract hand movement features
            hand_translation = moving_hand_frame[0]  # x, y, z
            hand_rotation = moving_hand_frame[1:]    # 3x3 rotation matrix
            
            # Map hand movement to joint changes
            # This is a simplified mapping - adjust based on your robot's kinematics
            num_joints = len(initial_pos)
            joint_changes = np.zeros(num_joints)
            
            # Map x, y, z translation to first 3 joints (if available)
            if num_joints >= 3:
                joint_changes[0] = hand_translation[0] * self.resolution_scale * 100  # shoulder_pan
                joint_changes[1] = hand_translation[1] * self.resolution_scale * 100  # shoulder_lift
                joint_changes[2] = hand_translation[2] * self.resolution_scale * 100  # elbow_flex
            
            # Map rotation to next 3 joints (if available)
            if num_joints >= 6:
                # Extract Euler angles from rotation matrix
                euler = Rotation.from_matrix(hand_rotation).as_euler('xyz')
                joint_changes[3] = euler[0] * self.resolution_scale * 50  # wrist_flex
                joint_changes[4] = euler[1] * self.resolution_scale * 50  # wrist_roll
                joint_changes[5] = euler[2] * self.resolution_scale * 50  # additional joint if available
            
            # Calculate new joint positions
            new_joint_positions = current_joint_positions[name] + torch.from_numpy(joint_changes).float()
            
            # Apply filter if enabled
            if self.use_filter:
                new_joint_positions = torch.from_numpy(
                    self.joint_filters[name](new_joint_positions.numpy())
                ).float()
            
            joint_position_changes[name] = new_joint_positions
        
        # Get gripper state
        gripper_state, status_change, gripper_flag = self.get_gripper_state_from_hand_keypoints()
        if gripper_flag == 1 and status_change is True:
            self.gripper_correct_state = gripper_state
        
        # Create action dictionary with joint positions
        action = {}
        for name, pos in joint_position_changes.items():
            action[name] = pos
        
        # Add gripper state
        action["gripper"] = self.gripper_correct_state
        
        return action

    def vr_teleop_step(self, record_data=False):
        """
        VR teleoperation step method that replaces the robot's teleop_step method.
        This method is called by the control_loop function.
        
        Args:
            record_data (bool): Whether to record data for dataset creation
            
        Returns:
            tuple: (observation_dict, action_dict) if record_data is True, None otherwise
        """
        # Make sure we're initialized
        if not self.initialized:
            success = self._initialize()
            if not success:
                logging.error("Failed to initialize VR teleoperator during teleop step")
                if record_data:
                    # Return empty observation and action
                    observation = self.robot.capture_observation()
                    action_dict = {"action": torch.zeros(1)}
                    return observation, action_dict
                return None
        
        # Check if we're still receiving fresh data
        if not self._keypoint_puller.is_data_fresh():
            logging.warning("No fresh data received, stopping robot movement")
            self.arm_teleop_state = ARM_TELEOP_STOP
            
            # If it's been a while since we received data, log a more serious warning
            if time.time() - self._keypoint_puller.last_receive_time > 5.0:
                logging.error("Connection to VR controller appears to be lost")
        
        # Get action from VR teleoperation
        action = self.apply_retargeted_angles()
        
        # If no valid action or paused, return None or empty dicts
        if action is None or self.arm_teleop_state == ARM_TELEOP_STOP:
            if record_data:
                # Create empty observation and action dictionaries
                observation = self.robot.capture_observation()
                # Create an empty action with the right dimensions
                empty_action = []
                for name in self.robot.follower_arms:
                    joint_pos = self.robot.follower_arms[name].read("Present_Position")
                    empty_action.append(torch.from_numpy(joint_pos))
                action_tensor = torch.cat(empty_action)
                action_dict = {"action": action_tensor}
                return observation, action_dict
            return None
        
        # Process the action for each arm
        for name in self.robot.follower_arms:
            if name in action:
                # Get the joint positions for this arm
                joint_positions = action[name]
                
                # Convert to numpy and send to the robot
                joint_positions_np = joint_positions.numpy().astype(np.float32)
                self.robot.follower_arms[name].write("Goal_Position", joint_positions_np)
        
        # Handle gripper separately if needed
        if "gripper" in action:
            gripper_state = action["gripper"]
            # In a real implementation, you would control the gripper here
            # For now, we'll just log it
            logging.info(f"Gripper state: {gripper_state}")
        
        # Log robot state to rerun if display is enabled
        if self.cfg.display_data:
            self._log_robot_state_to_rerun()
        
        # If recording data, return observation and action
        if record_data:
            observation = self.robot.capture_observation()
            
            # Create action tensor from all joint positions
            action_tensors = []
            for name in self.robot.follower_arms:
                if name in action:
                    action_tensors.append(action[name])
            
            action_tensor = torch.cat(action_tensors)
            action_dict = {"action": action_tensor}
            
            return observation, action_dict
        
        return None
    
    def _log_robot_state_to_rerun(self):
        """Log robot state to rerun for visualization."""
        if not self.initialized:
            return
            
        try:
            # Get joint positions for all arms
            joint_positions = []
            for name in self.robot.follower_arms:
                pos = self.robot.follower_arms[name].read("Present_Position")
                joint_positions.extend(pos)
            
            # Convert to numpy array
            joint_positions = np.array(joint_positions)
            
            # Log to rerun
            rr.log("robot/joint_positions", rr.Points3D(joint_positions))
            
            # Log gripper state
            rr.log("robot/gripper", rr.Scalar(self.gripper_correct_state))
            
            # Log VR controller state if available
            hand_frame = self._get_hand_frame()
            if hand_frame is not None:
                # Extract translation and rotation
                translation = hand_frame[0]
                rotation = Rotation.from_matrix(hand_frame[1:]).as_quat()
                
                # Log to rerun
                rr.log("vr/hand_pose", rr.Transform3D(
                    translation=translation,
                    rotation=rotation
                ))
        except Exception as e:
            logging.error(f"Error logging to rerun: {e}")


@safe_disconnect
def teleoperate_vr(robot: Robot, cfg: VRTeleoperateControlConfig):
    """
    Teleoperate a robot using VR controllers.
    
    Args:
        robot (Robot): Robot to teleoperate
        cfg (VRTeleoperateControlConfig): VR teleoperation configuration
    """
    #log_say("Starting VR teleoperation", cfg.play_sounds)
    
    # First connect to the robot
    if not robot.is_connected:
        robot.connect()
    
    # Initialize VR teleoperator
    teleoperator = VRTeleoperator(robot, cfg)
    
    # Log the actual port being used (which might be different from the configured port)
    logging.info(f"VR Teleoperation is listening on port {teleoperator.keypoint_port}")
    print(f"VR Teleoperation is listening on port {teleoperator.keypoint_port}")
    
    try:
        # Use the standard control_loop function
        control_loop(
            robot=robot,
            control_time_s=cfg.teleop_time_s,
            fps=cfg.fps,
            teleoperate=True,
            display_data=cfg.display_data,
        )
    except KeyboardInterrupt:
        logging.info("VR teleoperation interrupted by user")
    except Exception as e:
        logging.error(f"Error during VR teleoperation: {e}")
    finally:
        # Restore the original teleop_step method
        if hasattr(teleoperator, 'original_teleop_step'):
            robot.teleop_step = teleoperator.original_teleop_step
    
    #log_say("VR teleoperation complete", cfg.play_sounds)
