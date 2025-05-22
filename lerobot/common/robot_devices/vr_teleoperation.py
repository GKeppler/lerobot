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
import logging
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import rerun as rr
from scipy.spatial.transform import Rotation, Slerp
# from numpy.linalg import pinv # Not used

from lerobot.common.robot_devices.control_configs import VRTeleoperateControlConfig
from lerobot.common.robot_devices.control_utils import control_loop
from lerobot.common.robot_devices.utils import safe_disconnect # busy_wait not used
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.utils.utils import log_say


# Constants for VR teleoperation
ARM_TELEOP_STOP = 0
ARM_TELEOP_CONT = 1
ARM_HIGH_RESOLUTION = 0
ARM_LOW_RESOLUTION = 1
# SCALE_FACTOR = 1000.0  # Not used directly
BIMANUAL_VR_FREQ = 90  # Hz, Not directly used in this script logic

# Define the Oculus joints mapping
OCULUS_JOINTS = {
    'wrist': [0],
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20]
}


class Filter:
    def __init__(self, state, comp_ratio=0.6):
        # state is expected to be a 6D vector [pos_x, pos_y, pos_z, rot_rx, rot_ry, rot_rz]
        # This class is problematic if used for joint angles unless the robot has exactly 6 joints
        # and this specific filtering approach (treating first 3 as pos, next 3 as ori via Slerp) is intended.
        if len(state) < 6:
            raise ValueError(f"Filter expects at least a 6-element state, got {len(state)}")
        self.pos_state = state[:3]
        self.ori_state = state[3:6] # Assuming state[3:6] is a rotation vector
        self.comp_ratio = comp_ratio

    def __call__(self, next_state):
        if len(next_state) < 6:
             # If next_state is not 6D, return it unfiltered or handle error
            logging.warning(f"Filter called with state of length {len(next_state)}, expected >= 6. Returning unfiltered.")
            return next_state
            
        filtered_pos = self.pos_state * self.comp_ratio + next_state[:3] * (1 - self.comp_ratio)
        
        # Slerp for orientation part (next_state[3:6] must be a rotation vector)
        slerp_rotations = Rotation.from_rotvec(np.stack([self.ori_state, next_state[3:6]], axis=0))
        ori_interp = Slerp([0, 1], slerp_rotations)
        filtered_ori_rotvec = ori_interp([1 - self.comp_ratio])[0].as_rotvec()
        
        self.pos_state = filtered_pos
        self.ori_state = filtered_ori_rotvec
        
        # Reconstruct the full state vector
        # If original state had more than 6 elements, append the rest unfiltered
        filtered_state = np.concatenate([filtered_pos, filtered_ori_rotvec])
        if len(next_state) > 6:
            filtered_state = np.concatenate([filtered_state, next_state[6:]])
        return filtered_state


class ZMQKeypointPuller:
    """Puller for ZMQ keypoint data using PULL socket."""
    def __init__(self, port, max_retries=3, data_timeout=1.0):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        # Set a High Water Mark to prevent excessive buffering if the consumer (this script) is slow.
        # This means the sender (VR app) might block or drop messages if this side can't keep up.
        self.socket.set_hwm(10) 
        
        bound = False
        retry_count = 0
        self.port = port
        
        while not bound and retry_count < max_retries:
            try:
                bind_address = f"tcp://0.0.0.0:{self.port}"
                logging.info(f"Attempting to bind to {bind_address}")
                self.socket.bind(bind_address)
                bound = True
                logging.info(f"Successfully bound to {bind_address}")
            except zmq.error.ZMQError as e:
                retry_count += 1
                logging.warning(f"Failed to bind to port {self.port}: {e}")
                if retry_count < max_retries:
                    self.port += 1000
                    logging.info(f"Retrying with port {self.port}")
                    time.sleep(1)
                else:
                    logging.error(f"Failed to bind after {max_retries} retries")
                    raise
        
        self.message_data = {}
        self.message_timestamps = {}
        self.last_receive_time = time.time() 
        self.data_timeout = data_timeout

    def recv_keypoints_noblock(self):
        """ Non-blocking receive. Returns full message dict or None. Updates internal state. """
        try:
            message = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
            self.last_receive_time = time.time() # Update on any successful receive
            if isinstance(message, dict) and 'type' in message and 'data' in message:
                message_type = message['type']
                self.message_data[message_type] = message['data']
                self.message_timestamps[message_type] = time.time()
                return message 
            # logging.warning(f"Received non-standard or malformed ZMQ message: {message}")
            return None 
        except zmq.Again: 
            return None
        except Exception as e:
            logging.error(f"Error in ZMQ recv_keypoints_noblock: {e}")
            return None

    def get_latest_data(self, message_type):
        return self.message_data.get(message_type)
    
    def is_data_fresh(self, message_type=None):
        current_time = time.time()
        if message_type is None: # Check overall freshness
            return (current_time - self.last_receive_time) < self.data_timeout
        
        # Check freshness for a specific message type
        if message_type in self.message_timestamps:
            return (current_time - self.message_timestamps[message_type]) < self.data_timeout
        return False

    def close(self):
        logging.info(f"Closing ZMQ PULL socket on port {self.port}")
        if hasattr(self, 'socket') and self.socket and not self.socket.closed:
            self.socket.close(linger=0) # linger=0 ensures immediate close
        if hasattr(self, 'context') and self.context and not self.context.closed:
            self.context.term()


class VRTeleoperator:
    def __init__(self, robot: Robot, cfg: VRTeleoperateControlConfig):
        self.robot = robot
        self.cfg = cfg
        
        logging.info(f"Initializing ZMQ PULL socket for VR Teleoperation.")
        self._keypoint_puller = ZMQKeypointPuller(port=cfg.transformed_keypoint_port, data_timeout=cfg.vr_data_timeout if hasattr(cfg, 'vr_data_timeout') else 1.0)
        self.keypoint_port = self._keypoint_puller.port
        logging.info(f"VR Teleoperation PULL socket bound to port {self.keypoint_port}")
        
        self.gripper_flag = True  # Gripper state: True typically means open or 1.0
        self.pause_flag = True    # Control state: True means ARM_TELEOP_CONT (running)
        
        self.prev_pause_flag = not self.pause_flag # Initialize to ensure first check has a delta
        self.is_first_frame = True
        self.gripper_cnt = 0
        self.prev_gripper_flag = not self.gripper_flag # Initialize to ensure first check has a delta
        self.pause_cnt = 0
        self.gripper_correct_state = 1 if self.gripper_flag else 0 
        self.resolution_scale = 1.0 
        self.arm_teleop_state = ARM_TELEOP_STOP # Start stopped; will transition on first valid frame/unpause
        
        self.initial_joint_positions = {} # Stores robot joint state at the time of _reset_teleop
        self.hand_init_frame = None       # Stores hand frame at the time of _reset_teleop
        self.joint_filters = {}
        self.use_filter = cfg.use_filter
        self.initialized = False
        self.shutdown_requested = False
        
        self.original_teleop_step = robot.teleop_step
        robot.teleop_step = self.vr_teleop_step
        
        logging.info("VR Teleoperator initialized")

    def _process_incoming_messages(self):
        """Helper to process all available messages from the puller and update state."""
        try:
            while True: 
                msg_dict = self._keypoint_puller.recv_keypoints_noblock()
                if msg_dict is None: 
                    break 
                if msg_dict.get('type') == "shutdown":
                    logging.info("Shutdown message received from VR controller.")
                    self.shutdown_requested = True
        except Exception as e:
            logging.error(f"Error processing incoming ZMQ messages: {e}")

    def _get_hand_frame(self):
        # _process_incoming_messages() # Called by parent function like apply_retargeted_angles
        data = self._keypoint_puller.get_latest_data('transformed_hand_frame')
        if data is None:
            return None
        return np.asanyarray(data).reshape(4, 3)
    
    def _get_resolution_scale_mode(self):
        # _process_incoming_messages()
        data = self._keypoint_puller.get_latest_data('button')
        if data is None:
            return ARM_HIGH_RESOLUTION 
        return np.asanyarray(data).reshape(1)[0]
    
    def _get_arm_teleop_state_from_keypoints(self):
        # This method determines the desired teleop state based on finger gestures
        # It does NOT call _process_incoming_messages; assumes parent caller does.
        pause_bool_state, pause_status_changed = self.get_pause_state_from_keypoints(process_msg=False) # New: expects 2
        # pause_bool_state is True if fingers indicate "continue", False if "pause"
        
        current_teleop_state_gesture = ARM_TELEOP_CONT if pause_bool_state else ARM_TELEOP_STOP
            
        return current_teleop_state_gesture, pause_status_changed

    def _initialize(self):
        if not self.robot.is_connected:
            logging.warning("Robot is not connected. Cannot initialize VR teleoperator.")
            return False
        try:
            self.initial_joint_positions = {}
            for name in self.robot.follower_arms:
                joint_pos = self.robot.follower_arms[name].read("Present_Position")
                self.initial_joint_positions[name] = torch.from_numpy(joint_pos)
            
            # Filter initialization:
            # The provided Filter class is for 6D Cartesian states.
            # Applying it to joint angles is only appropriate if the robot has exactly 6 joints
            # and this specific filtering (Slerp on last 3 elements) is desired for those joints.
            # If cfg.use_filter is True, we attempt to initialize.
            if self.use_filter:
                self.joint_filters = {} # Clear previous
                for name, pos_tensor in self.initial_joint_positions.items():
                    pos_numpy = pos_tensor.numpy()
                    if len(pos_numpy) >= 6:
                        try:
                            self.joint_filters[name] = Filter(pos_numpy.copy(), comp_ratio=self.cfg.filter_ratio)
                        except ValueError as e:
                             logging.warning(f"Could not initialize Filter for arm '{name}' (requires >=6 values, got {len(pos_numpy)}): {e}. Filtering disabled for this arm.")
                    else:
                        logging.warning(f"Filter not applied to arm '{name}': requires at least 6 values, got {len(pos_numpy)}. Filtering disabled for this arm.")
            
            self.initialized = True
            logging.info("VR teleoperator initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Error initializing VR teleoperator: {e}", exc_info=True)
            self.initialized = False 
            return False

    def _reset_teleop(self):
        logging.info('****** RESETTING TELEOP ****** ')
        if not self.initialized:
            if not self._initialize(): # Try to initialize if not already
                logging.error("Failed to initialize VR teleoperator during reset. Aborting reset.")
                return
        
        # Update initial joint positions for the robot
        new_initial_joint_positions = {}
        for name in self.robot.follower_arms:
            try:
                joint_pos = self.robot.follower_arms[name].read("Present_Position")
                new_initial_joint_positions[name] = torch.from_numpy(joint_pos)
            except Exception as e:
                logging.error(f"Could not read joint positions for arm {name} during reset: {e}")
                # Keep old initial position if read fails, or handle error more gracefully
                if name in self.initial_joint_positions:
                    new_initial_joint_positions[name] = self.initial_joint_positions[name]
                else: # Critical failure for this arm
                    logging.error(f"Cannot proceed with reset for arm {name} without initial position.")
                    self.arm_teleop_state = ARM_TELEOP_STOP # Safety
                    return
        self.initial_joint_positions = new_initial_joint_positions

        # Re-initialize filters with new initial positions if filter is enabled
        if self.use_filter:
            self.joint_filters = {} 
            for name, pos_tensor in self.initial_joint_positions.items():
                pos_numpy = pos_tensor.numpy()
                if len(pos_numpy) >= 6 :
                    try:
                        self.joint_filters[name] = Filter(pos_numpy.copy(), comp_ratio=self.cfg.filter_ratio)
                    except ValueError as e:
                        logging.warning(f"Could not re-initialize Filter for arm '{name}' during reset: {e}. Filtering disabled for this arm.")
                else:
                     logging.warning(f"Filter not applied to arm '{name}' during reset: requires at least 6 values, got {len(pos_numpy)}. Filtering disabled for this arm.")

        # Get initial hand frame (make sure messages are processed first)
        self._process_incoming_messages()
        first_hand_frame = self._get_hand_frame()
        retry_count = 0
        max_retries = int(5 / 0.1) # Wait up to 5 seconds, checking every 0.1s
        while first_hand_frame is None and retry_count < max_retries and not self.shutdown_requested:
            logging.info("Waiting for initial hand frame to reset teleop...")
            time.sleep(0.1)
            self._process_incoming_messages() # Process messages again before getting data
            first_hand_frame = self._get_hand_frame()
            retry_count += 1
        
        if first_hand_frame is None:
            logging.error("Failed to get initial hand frame for reset. Teleop might not function correctly.")
            self.is_first_frame = True 
            self.arm_teleop_state = ARM_TELEOP_STOP # Safety stop
            return

        self.hand_init_frame = first_hand_frame
        self.is_first_frame = False
        logging.info("Resetting teleop complete")

    def get_gripper_state_from_keypoints(self, process_msg=True):
        if process_msg: self._process_incoming_messages()
        transformed_hand_coords = self._keypoint_puller.get_latest_data('transformed_hand_coords')
        
        if transformed_hand_coords is None:
            return self.gripper_flag, False # current state, did not change status

        distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['pinky'][-1]] - transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
        thresh = 0.03
        toggled_this_step = False
        if distance < thresh:
            self.gripper_cnt += 1
            if self.gripper_cnt == 1: 
                self.prev_gripper_flag = self.gripper_flag # Store before toggle
                self.gripper_flag = not self.gripper_flag 
                toggled_this_step = True
        else: 
            self.gripper_cnt = 0 
        
        status_changed = toggled_this_step # True if state was toggled *this specific step*
        return self.gripper_flag, status_changed
   
    def get_pause_state_from_keypoints(self, process_msg=True):
        if process_msg: self._process_incoming_messages()
        transformed_hand_coords = self._keypoint_puller.get_latest_data('transformed_hand_coords')

        if transformed_hand_coords is None:
            return self.pause_flag, False # current state, did not change status

        ring_distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['ring'][-1]] - transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
        middle_distance = np.linalg.norm(transformed_hand_coords[OCULUS_JOINTS['middle'][-1]] - transformed_hand_coords[OCULUS_JOINTS['thumb'][-1]])
        thresh = 0.03 
        toggled_this_step = False
        if ring_distance < thresh or middle_distance < thresh:
            self.pause_cnt += 1
            if self.pause_cnt == 1: 
                self.prev_pause_flag = self.pause_flag # Store before toggle
                self.pause_flag = not self.pause_flag 
                toggled_this_step = True
        else:
            self.pause_cnt = 0 
        
        status_changed = toggled_this_step # True if state was toggled *this specific step*
        return self.pause_flag, status_changed

    def apply_retargeted_angles(self):
        # This method is called after _process_incoming_messages has run in vr_teleop_step
        if not self.initialized:
            if not self._initialize():
                logging.error("Cannot apply_retargeted_angles: VR teleoperator not initialized.")
                return None
        
        # Determine desired teleop state from gestures
        # Note: self.shutdown_requested is already updated by _process_incoming_messages
        desired_teleop_state_gesture, pause_status_changed = self._get_arm_teleop_state_from_keypoints()

        if self.is_first_frame or \
           (self.arm_teleop_state == ARM_TELEOP_STOP and desired_teleop_state_gesture == ARM_TELEOP_CONT):
            if not self.shutdown_requested: 
                self._reset_teleop()
                if self.is_first_frame: # If _reset_teleop failed to get hand_init_frame
                    self.arm_teleop_state = ARM_TELEOP_STOP # Stay stopped
                    return None
            else: # Shutdown requested during this initial phase
                self.is_first_frame = False 
                self.arm_teleop_state = ARM_TELEOP_STOP
                return None

        self.arm_teleop_state = desired_teleop_state_gesture
        if self.shutdown_requested: # Shutdown overrides gesture
            self.arm_teleop_state = ARM_TELEOP_STOP
        
        if self.arm_teleop_state == ARM_TELEOP_STOP:
            return None 

        # --- Robot is running: ARM_TELEOP_CONT ---
        arm_teleoperation_scale_mode = self._get_resolution_scale_mode() 
        if arm_teleoperation_scale_mode == ARM_HIGH_RESOLUTION:
            self.resolution_scale = 1.0
        elif arm_teleoperation_scale_mode == ARM_LOW_RESOLUTION:
            self.resolution_scale = 0.6

        moving_hand_frame = self._get_hand_frame() 
        if moving_hand_frame is None or self.hand_init_frame is None:
            # If hand_init_frame is None, reset probably failed.
            logging.warning("apply_retargeted_angles: Missing hand frame data. Stopping movement.")
            self.arm_teleop_state = ARM_TELEOP_STOP 
            return None

        # --- Calculate Action ---
        current_robot_joint_positions = {}
        for name in self.robot.follower_arms:
            current_robot_joint_positions[name] = torch.from_numpy(self.robot.follower_arms[name].read("Present_Position"))
        
        action_output_joint_positions = {}
        
        # --- Hand to Joint Mapping (Preserved from original, with deltas relative to init hand pose) ---
        # This maps hand pose (relative to its pose at teleop reset) to robot joint *increments*.
        # These increments are then added to the *current* robot joint positions.
        for name, initial_robot_joints_at_reset in self.initial_joint_positions.items():
            hand_trans_abs = moving_hand_frame[0]
            hand_rot_abs_mat = moving_hand_frame[1:]

            # Calculate hand deltas relative to the hand pose at the last _reset_teleop
            hand_trans_delta = hand_trans_abs - self.hand_init_frame[0]
            
            R_init_hand = Rotation.from_matrix(self.hand_init_frame[1:])
            R_current_hand = Rotation.from_matrix(hand_rot_abs_mat)
            R_delta_hand = R_init_hand.inv() * R_current_hand
            euler_delta_hand = R_delta_hand.as_euler('xyz') # More stable for delta mapping

            num_joints = len(initial_robot_joints_at_reset)
            joint_deltas_to_apply = np.zeros(num_joints)
            
            # Heuristic mapping from hand deltas to joint deltas
            # Adjust scaling factors (100, 50) as needed for your robot's sensitivity
            if num_joints >= 1: joint_deltas_to_apply[0] = hand_trans_delta[0] * self.resolution_scale * 100 
            if num_joints >= 2: joint_deltas_to_apply[1] = hand_trans_delta[1] * self.resolution_scale * 100 
            if num_joints >= 3: joint_deltas_to_apply[2] = hand_trans_delta[2] * self.resolution_scale * 100
            
            if num_joints >= 6: # Assuming joints 3,4,5 map to euler X,Y,Z changes
                joint_deltas_to_apply[3] = euler_delta_hand[0] * self.resolution_scale * 50 
                joint_deltas_to_apply[4] = euler_delta_hand[1] * self.resolution_scale * 50 
                joint_deltas_to_apply[5] = euler_delta_hand[2] * self.resolution_scale * 50 
            
            # New target joint positions = current robot joints + calculated deltas
            # This means the robot moves relative to its current pose based on how far the hand
            # has moved from its initial pose at reset.
            target_joint_positions = current_robot_joint_positions[name] + torch.from_numpy(joint_deltas_to_apply).float()

            if self.use_filter and name in self.joint_filters:
                try:
                    target_joint_positions = torch.from_numpy(
                        self.joint_filters[name](target_joint_positions.numpy().copy()) # Pass copy
                    ).float()
                except Exception as e:
                    logging.warning(f"Error applying filter to arm {name}: {e}. Using unfiltered values.")

            action_output_joint_positions[name] = target_joint_positions
        
        # --- Gripper State ---
        gripper_bool_state, gripper_status_changed = self.get_gripper_state_from_keypoints(process_msg=False)
        if gripper_status_changed: 
            self.gripper_correct_state = 1 if gripper_bool_state else 0
        
        # --- Assemble Action Dictionary ---
        final_action = {}
        for name, pos in action_output_joint_positions.items():
            final_action[name] = pos
        final_action["gripper"] = self.gripper_correct_state 
        
        return final_action

    def vr_teleop_step(self, record_data=False):
        if not self.initialized:
            if not self._initialize():
                logging.error("Failed to initialize VR teleoperator during teleop step")
                if record_data:
                    observation = self.robot.capture_observation() if self.robot.is_connected else {}
                    # Determine action dimension safely
                    act_dim = 1
                    if hasattr(self.robot, "action_space") and self.robot.action_space and hasattr(self.robot.action_space, "shape"):
                         act_dim = self.robot.action_space.shape[0]
                    elif self.robot.follower_arms: # Fallback to number of joints of first arm if possible
                        first_arm_name = next(iter(self.robot.follower_arms))
                        try:
                            act_dim = len(self.robot.follower_arms[first_arm_name].read("Present_Position"))
                        except: pass # Keep act_dim = 1
                            
                    action_dict = {"action": torch.zeros(act_dim), "gripper": torch.tensor([self.gripper_correct_state], dtype=torch.float32)}
                    return observation, action_dict, {}
                return None

        self._process_incoming_messages() # Process ZMQ, potentially sets self.shutdown_requested

        action_to_execute = None # Default to no action

        # Check for explicit shutdown request or critical data loss leading to shutdown
        if self.shutdown_requested:
            logging.info("VR teleop step: Shutdown requested. Halting robot.")
            self.arm_teleop_state = ARM_TELEOP_STOP
        elif not self._keypoint_puller.is_data_fresh(None) and \
             (time.time() - self._keypoint_puller.last_receive_time > (self._keypoint_puller.data_timeout + 4.0)): # Extended timeout
            logging.error(f"VR teleop step: Connection to VR controller appears lost (timeout > {self._keypoint_puller.data_timeout + 4.0}s). Forcing shutdown.")
            self.shutdown_requested = True # Escalate to full shutdown
            self.arm_teleop_state = ARM_TELEOP_STOP
        
        # If not actively shutting down, call apply_retargeted_angles.
        # This method will handle state transitions (STOP <-> CONT) based on gestures/reset logic
        # and will update self.arm_teleop_state. It returns an action or None.
        if not self.shutdown_requested:
            action_to_execute = self.apply_retargeted_angles()
        
        # After apply_retargeted_angles, self.arm_teleop_state reflects the current desired state.
        # If action_to_execute is None or state is STOP, then no motion.
        if action_to_execute is None or self.arm_teleop_state == ARM_TELEOP_STOP:
            if self.arm_teleop_state == ARM_TELEOP_STOP and not self.shutdown_requested:
                 # This log can be noisy if frequently paused, use logging.debug or remove if too verbose
                 # logging.info("Robot is paused or waiting for initial valid data to start/resume.")
                 pass
            
            if record_data: # If recording, provide current state as "action" (hold position)
                observation = self.robot.capture_observation() if self.robot.is_connected else {}
                joint_values_for_action = []
                act_dim = 1 # Default action dimension
                if self.robot.is_connected:
                    if self.robot.follower_arms:
                        first_arm_name = next(iter(self.robot.follower_arms))
                        try:
                             # Get full action dim from all follower arms' joints
                            current_joint_dims = 0
                            for name in self.robot.follower_arms:
                                joint_pos = self.robot.follower_arms[name].read("Present_Position")
                                joint_values_for_action.append(torch.from_numpy(joint_pos))
                                current_joint_dims += len(joint_pos)
                            if current_joint_dims > 0: act_dim = current_joint_dims
                        except Exception as e:
                            logging.error(f"Error reading Present_Position for arms (record_data stop): {e}")
                            if hasattr(self.robot, "action_space") and self.robot.action_space and hasattr(self.robot.action_space, "shape"):
                                act_dim = self.robot.action_space.shape[0] # Fallback to configured action space
                    elif hasattr(self.robot, "action_space") and self.robot.action_space and hasattr(self.robot.action_space, "shape"):
                        act_dim = self.robot.action_space.shape[0]


                action_tensor = torch.cat(joint_values_for_action) if joint_values_for_action else torch.zeros(act_dim, dtype=torch.float32)
                action_dict_to_return = {"action": action_tensor, "gripper": torch.tensor([self.gripper_correct_state], dtype=torch.float32)}
                return observation, action_dict_to_return
            return None # No action to send if not recording and stopped/paused

        # If we reach here, action_to_execute is not None and self.arm_teleop_state is ARM_TELEOP_CONT
        
        # Process the action for each arm
        for name in self.robot.follower_arms:
            if name in action_to_execute and self.robot.is_connected:
                joint_positions = action_to_execute[name]
                joint_positions_np = joint_positions.numpy().astype(np.float32)
                try:
                    self.robot.follower_arms[name].write("Goal_Position", joint_positions_np)
                except Exception as e:
                    logging.error(f"Error writing Goal_Position for {name}: {e}")
        
        if "gripper" in action_to_execute and self.robot.is_connected:
            gripper_cmd = action_to_execute["gripper"] 
            # logging.info(f"Gripper command: {gripper_cmd}") # Implement actual gripper control
            # self.robot.set_gripper(float(gripper_cmd)) # Example

        if self.cfg.display_data:
            self._log_robot_state_to_rerun()
        
        if record_data: # If recording, format and return observation and executed action
            observation = self.robot.capture_observation() if self.robot.is_connected else {}
            
            recorded_action_tensors = []
            act_dim_final = 1 # Default
            if self.robot.is_connected:
                current_joint_dims_rec = 0
                for name in self.robot.follower_arms:
                    if name in action_to_execute: 
                        recorded_action_tensors.append(action_to_execute[name])
                        current_joint_dims_rec += len(action_to_execute[name])
                if current_joint_dims_rec > 0: act_dim_final = current_joint_dims_rec
                elif hasattr(self.robot, "action_space") and self.robot.action_space and hasattr(self.robot.action_space, "shape"):
                     act_dim_final = self.robot.action_space.shape[0]

            final_action_tensor = torch.cat(recorded_action_tensors) if recorded_action_tensors else torch.zeros(act_dim_final, dtype=torch.float32)
            action_dict_to_return = {
                "action": final_action_tensor, 
                "gripper": torch.tensor([action_to_execute.get("gripper", self.gripper_correct_state)], dtype=torch.float32)
            }
            return observation, action_dict_to_return
        
        return None

    def _log_robot_state_to_rerun(self):
        if not self.initialized or not rr or not self.robot.is_connected or not self.cfg.display_data:
            return
        try:
            robot_name_prefix = self.robot.name if hasattr(self.robot, "name") else "robot"
            current_time_ns = int(time.time() * 1e9) # For Rerun time
            rr.set_time_nanos("sim_time", current_time_ns)

            all_joint_positions_dict = {}
            for arm_idx, name in enumerate(self.robot.follower_arms):
                pos = self.robot.follower_arms[name].read("Present_Position")
                for joint_idx, p_val in enumerate(pos):
                     all_joint_positions_dict[f"arm{arm_idx}/joint{joint_idx}"] = p_val
            if all_joint_positions_dict:
                 rr.log(f"{robot_name_prefix}/joint_positions", rr. MuitosScalars(all_joint_positions_dict))
            
            rr.log(f"{robot_name_prefix}/gripper_state", rr.Scalar(self.gripper_correct_state))
            
            hand_frame_data = self._keypoint_puller.get_latest_data('transformed_hand_frame')
            if hand_frame_data is not None:
                hand_frame_np = np.asanyarray(hand_frame_data).reshape(4,3)
                translation = hand_frame_np[0]
                # The hand_frame[1:] is R (cols are basis vectors of new frame in old frame)
                # Rerun Transform3D expects mat3x3 for rotation where columns are basis vectors.
                rotation_matrix = hand_frame_np[1:] 
                rr.log("vr_controller/hand_pose", rr.Transform3D(translation=translation, mat3x3=rotation_matrix))
        except Exception as e:
            logging.error(f"Error logging to rerun: {e}", exc_info=True)


@safe_disconnect
def teleoperate_vr(robot: Robot, cfg: VRTeleoperateControlConfig):
    # #log_say("Starting VR teleoperation", cfg.play_sounds)
    
    if not robot.is_connected:
        try:
            robot.connect()
        except Exception as e:
            logging.error(f"Failed to connect to robot: {e}")
            # #log_say("Failed to connect to robot.", cfg.play_sounds)
            return 

    teleoperator = VRTeleoperator(robot, cfg)
    
    logging.info(f"VR Teleoperation PULL socket is listening on TCP port {teleoperator.keypoint_port} (all interfaces: 0.0.0.0)")
    print(f"VR Teleoperation PULL socket is listening on TCP port {teleoperator.keypoint_port} (all interfaces: 0.0.0.0)")
    
    try:
        control_loop(
            robot=robot,
            control_time_s=cfg.teleop_time_s,
            fps=cfg.fps,
            teleoperate=True, 
            display_data=cfg.display_data,
        )
    except KeyboardInterrupt:
        logging.info("VR teleoperation interrupted by user.")
    except Exception as e:
        logging.error(f"Unhandled error during VR teleoperation: {e}", exc_info=True)
    finally:
        logging.info("Cleaning up VR teleoperation resources...")
        # #log_say("Stopping VR teleoperation.", cfg.play_sounds)
        if hasattr(teleoperator, 'shutdown_requested'): # Signal shutdown to teleoperator loop if not already
            teleoperator.shutdown_requested = True 
            time.sleep(1/cfg.fps + 0.1) # Give one loop cycle chance to process shutdown

        if hasattr(teleoperator, 'original_teleop_step') and robot:
            robot.teleop_step = teleoperator.original_teleop_step
        if hasattr(teleoperator, '_keypoint_puller'):
            teleoperator._keypoint_puller.close()
        # Robot disconnection is typically handled by @safe_disconnect decorator
        # or by the script that calls this function.
    
    logging.info("VR teleoperation complete.")
    # #log_say("VR teleoperation complete.", cfg.play_sounds)