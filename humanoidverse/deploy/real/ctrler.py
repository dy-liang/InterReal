






from typing import Union, List, Tuple
import numpy as np
import time
import torch
import signal

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize # type: ignore
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize # type: ignore
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_ # type: ignore
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_ # type: ignore    
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG # type: ignore
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo # type: ignore
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG # type: ignore
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo # type: ignore
from unitree_sdk2py.utils.crc import CRC # type: ignore

from humanoidverse.utils.real.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from humanoidverse.utils.real.rotation_helper import get_gravity_orientation, transform_imu_data, quaternion_to_euler_array
from humanoidverse.utils.real.remote_controller import RemoteController, KeyMap
from humanoidverse.utils.helpers import parse_observation
from humanoidverse.data.robots.dtype import RobotExitException

import math
from typing import Dict
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
from omegaconf import OmegaConf
from loguru import logger
import os
import traceback

from ..urcirobot import URCIRobot
np2torch = lambda x: torch.tensor(x, dtype=torch.float32)
torch2np = lambda x: x.cpu().numpy()
def wrap_to_pi_float(angles:float):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles


class LowLevelMagic:
    low_state: LowStateHG
    low_cmd: LowCmdHG
    joystick: RemoteController
    
    _tick:int = 0
    _is_alive = True
    def __init__(self, cfg: OmegaConf):
        
        def signal_handler(sig, frame):
            logger.info("Ctrl+C  Exiting safely...")
            self.safe_exit()
        signal.signal(signal.SIGINT, signal_handler)
        
        config = cfg.deploy
        self.joystick = RemoteController()
        ChannelFactoryInitialize(0, config.net)
        
        
        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = config.mode_machine
            # self.mode_machine_ = 0
            # breakpoint() # check the mode machine

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.receive_state_handler, 10)

        elif config.msg_type == "go":
            raise ValueError("Not implemented for go msg type. This code is designed for hg msg type.")
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        # self._wait() #DEBUG:

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=config.weak_motor)
            
    def safe_exit(self):
        logger.info("Real Robot Exiting...")
        # os._exit(1)
        self._is_alive = False
        raise RobotExitException("Real Robot Exiting")
        
    def sanity_check(self):
        # print("Don't forget to check the New Exit Logic.");breakpoint()
        if not self._is_alive:
            self.safe_exit()
        for motor_idx in range(len(self.low_cmd.motor_cmd)):
            if ((np.abs(self.low_cmd.motor_cmd[motor_idx].q - self.low_state.motor_state[motor_idx].q) > 3 ) or 
                (np.abs(self.low_cmd.motor_cmd[motor_idx].dq) > 25)):

                logger.error(f"Action of joint {motor_idx} is too large.\n"
                             f"target q\t: {self.low_cmd.motor_cmd[motor_idx].q} \n"
                             f"target dq\t: {self.low_cmd.motor_cmd[motor_idx].dq} \n"
                             f"q\t\t: {self.low_state.motor_state[motor_idx].q} \n"
                             f"dq\t\t: {self.low_state.motor_state[motor_idx].dq}\n")
                self.safe_exit()
        
    # Entry point: 
    #   - send_cmd, 
    #   - receive_state_handler
    # 
        
    def send_cmd(self):
        self.sanity_check()
        self.low_cmd.crc = CRC().Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def LowStateCallback(self):
        
        if self.joystick.button[KeyMap.B] == 1: # quick stop
            self.safe_exit()
        pass
    
    def receive_state_handler(self, msg: LowStateHG):
        if not self._is_alive:
            return
        try:
            self.low_state = msg
            self.mode_machine_ = self.low_state.mode_machine
            self.joystick.set(self.low_state.wireless_remote)
            self._tick = self.low_state.tick
            
            self.sanity_check()
            self.LowStateCallback()
        except Exception as e:
            print("Receive State Handler: Exception >>> ", e)
            self.safe_exit()
        
    def _wait(self):
        while self.low_state.tick == 0:
            time.sleep(0.1)
            logger.info("Waiting for the robot to connect...")
        logger.info("Successfully connected to the robot.")
        

    # @dataclass
    # @annotate.final
    # @annotate.autoid("sequential")
    # class LowCmd_(idl.IdlStruct, typename="unitree_hg.msg.dds_.LowCmd_"):
    #     mode_pr: types.uint8
    #     mode_machine: types.uint8
    #     motor_cmd: types.array['unitree_sdk2py.idl.unitree_hg.msg.dds_.MotorCmd_', 35]
    #     reserve: types.array[types.uint32, 4]
    #     crc: types.uint32
    
    
    # @dataclass
    # @annotate.final
    # @annotate.autoid("sequential")
    # class MotorCmd_(idl.IdlStruct, typename="unitree_hg.msg.dds_.MotorCmd_"):
    #     mode: types.uint8
    #     q: types.float32
    #     dq: types.float32
    #     tau: types.float32
    #     kp: types.float32
    #     kd: types.float32
    #     reserve: types.uint32
    
    # 
    # @dataclass
    # @annotate.final
    # @annotate.autoid("sequential")
    # class LowState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.LowState_"):
    #     version: types.array[types.uint32, 2]
    #     mode_pr: types.uint8
    #     mode_machine: types.uint8
    #     tick: types.uint32
    #     imu_state: 'unitree_sdk2py.idl.unitree_hg.msg.dds_.IMUState_'
    #     motor_state: types.array['unitree_sdk2py.idl.unitree_hg.msg.dds_.MotorState_', 35]
    #     wireless_remote: types.array[types.uint8, 40]
    #     reserve: types.array[types.uint32, 4]
    #     crc: types.uint32


    # @dataclass
    # @annotate.final
    # @annotate.autoid("sequential")
    # class MotorState_(idl.IdlStruct, typename="unitree_hg.msg.dds_.MotorState_"):
    #     mode: types.uint8
    #     q: types.float32
    #     dq: types.float32
    #     ddq: types.float32
    #     tau_est: types.float32
    #     temperature: types.array[types.int16, 2]
    #     vol: types.float32
    #     sensor: types.array[types.uint32, 2]
    #     motorstate: types.uint32
    #     reserve: types.array[types.uint32, 4]



class RealRobot(URCIRobot, LowLevelMagic):
    REAL=True
    num_real_dofs: int = 29
    def __init__(self, cfg: OmegaConf):
        assert self.num_real_dofs == 29, "Only 29 dofs are supported for now"
        
        self.dof_idx_23_to_29: List[int] = cfg.deploy.dof_idx_23_to_29
        self.dof_idx_locked: List[int] = [i for i in range(0, 29) if i not in self.dof_idx_23_to_29]
        self.locked_kp: float = cfg.deploy.locked_kp
        self.locked_kd: float = cfg.deploy.locked_kd
        
        
        URCIRobot.__init__(self, cfg)
        LowLevelMagic.__init__(self, cfg)
        
        
        
        logger.info("Initializing **Real** Robot")
        logger.info("Task Name: {}".format(cfg.log_task_name))
        logger.info("Robot Type: {}".format(cfg.robot.asset.robot_type))
        
        self.Reset()
    
    def _make_buffer(self):
        super()._make_buffer()
        
        self.q_real = np.zeros(self.num_real_dofs)
        self.dq_real = np.zeros(self.num_real_dofs)
        self.ddq_real = np.zeros(self.num_real_dofs)
        self.tau_est = np.zeros(self.num_real_dofs)

        self.kp_real = np.zeros(self.num_real_dofs)
        self.kd_real = np.zeros(self.num_real_dofs)
        
        self.dof_init_pose_real = np.zeros(self.num_real_dofs)
        
        # 填充有效关节的kp/kd
        for i, j in enumerate(self.dof_idx_23_to_29):
            self.kp_real[j] = self.kp[i]
            self.kd_real[j] = self.kd[i]
            self.dof_init_pose_real[j] = self.dof_init_pose[i]
            
        # 填充锁定关节的kp/kd
        for j in self.dof_idx_locked:
            self.kp_real[j] = self.locked_kp
            self.kd_real[j] = self.locked_kd
    
    def GetState(self):
        for j in range(self.num_real_dofs):
            self.q_real[j] = self.low_state.motor_state[j].q
            self.dq_real[j] = self.low_state.motor_state[j].dq
            self.ddq_real[j] = self.low_state.motor_state[j].ddq
            self.tau_est[j] = self.low_state.motor_state[j].tau_est
            
        # print('DEBUG: q',self.q_real)
            
        self.q = self.q_real[self.dof_idx_23_to_29]
        self.dq = self.dq_real[self.dof_idx_23_to_29] 
        # breakpoint() # check the grammar
         
        # quat, omega
        
        self.quat = np.array(self.low_state.imu_state.quaternion)[[1,2,3,0]] # WXYZ to XYZW
        self.omega = np.array(self.low_state.imu_state.gyroscope)
        # self.omega = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        self.gvec = get_gravity_orientation(self.quat[[3,0,1,2]]) # get_gravity_orientation need WXYZ
        # assert self.quat[[3,0,1,2]] == np.array(self.low_state.imu_state.quaternion), "quat mismatch"
        
        self.rpy = np.array(self.low_state.imu_state.rpy)
        self.rpy[self.rpy > math.pi] -= 2 * math.pi
        
        
        self.cmd[0] = self.joystick.ly *0.5
        self.cmd[1] = self.joystick.lx * -1 * 0.5
        self.cmd[3] += self.joystick.rx * -1 * self.dt * 0.5
        
        if self.heading_cmd:
            self.cmd[2] = np.clip(0.5*wrap_to_pi_float(self.cmd[3] - self.rpy[2]), -1., 1.)
        else:
            self.cmd[2] = self.joystick.rx * -1 * 0.3
            
        if self.is_motion_tracking:
            self.motion_time = (self.timer) * self.dt 
            self.ref_motion_phase = self.motion_time / self.motion_len
            
        print("cmd: ", self.cmd,end='\r\b')
        # print("cmd: ", self.cmd,'| rpy: ', self.rpy, end='\r\b')
        # breakpoint() # check the syntax of self.cmd
        # TODO: add real-time logging for the state
    
    def _reset(self):
        # self.ToZeroTorque()
        
        self.ToDamping()
        self.MoveToDefaultPose()
        self.KeepDefaultPose()
        
    def _apply_action(self, target_q):
        self.SetPose(target_q)
        
    
    def _get_motion_to_save_torch(self)->Tuple[float, Dict[str, torch.Tensor]]:
        
        from scipy.spatial.transform import Rotation as sRot
        
        motion_time = (self.timer) * self.dt 

        root_rot = self.quat # XYZW
        root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot).as_rotvec()).float() # type: ignore
        dof = self.q
        # T, num_env, J, 3
        # print(self._motion_lib.mesh_parsers.dof_axis)
        pose_aa = torch.cat([root_rot_vec[..., None, :], 
                                torch.from_numpy(self._dof_axis * dof[..., None]), 
                                torch.zeros((self.num_augment_joint, 3))], axis = 0) # type: ignore
        
        return motion_time, {
            'root_trans_offset': torch.tensor([0,0,0.8], dtype=torch.float32),
            'root_rot': torch.from_numpy(root_rot), # 统一save xyzw
            'dof': torch.from_numpy(dof).clone(),
            'pose_aa': pose_aa,
            'action': torch.from_numpy(self.act).clone(),
            'actor_obs': np2torch(self.obs_buf_dict['actor_obs']),
            'terminate': torch.zeros((1,)),
            'dof_vel': torch.from_numpy(self.dq).clone(),
            'root_lin_vel': torch.zeros((1, 3)),
            'root_ang_vel': torch.from_numpy(self.omega).clone(),
            
            
            'clock_time': torch.tensor(time.time(), dtype=torch.float32),
            # 'clock_time': torch.from_numpy(np.array([time.time()])),
            'tau': torch.from_numpy(self.tau_est).clone(),
            'cmd': torch.from_numpy(self.cmd).clone(),
            'q_real': torch.from_numpy(self.q_real).clone(),
            'dq_real': torch.from_numpy(self.dq_real).clone(),
            'ddq_real': torch.from_numpy(self.ddq_real).clone(),
            'accelerometer': torch.tensor(self.low_state.imu_state.accelerometer).clone(),
            
        }
        
        
    def _get_motion_to_save_np(self)->Tuple[float, Dict[str, np.ndarray]]:
        
        from scipy.spatial.transform import Rotation as sRot
        
        motion_time = (self.timer) * self.dt 

        root_rot = self.quat # XYZW
        root_rot_vec = np.array(sRot.from_quat(root_rot).as_rotvec(), dtype=np.float32) # type: ignore
        dof = self.q
        # T, num_env, J, 3
        # print(self._motion_lib.mesh_parsers.dof_axis)
        pose_aa = torch.cat([root_rot_vec[..., None, :], 
                                torch.from_numpy(self._dof_axis * dof[..., None]), 
                                torch.zeros((self.num_augment_joint, 3))], axis = 0) # type: ignore
        
        raise NotImplementedError("This code is not tested yet")
        return motion_time, {
            'root_trans_offset': np.array([0,0,0.8], dtype=np.float32),
            'root_rot': root_rot, # 统一save xyzw
            'dof': dof.copy(),
            'pose_aa': pose_aa,
            'action': self.act.copy(),
            'actor_obs': self.obs_buf_dict['actor_obs'],
            'terminate': np.zeros((1,)),
            'dof_vel':self.dq.copy(),
            'root_lin_vel': np.zeros((1, 3)),
            'root_ang_vel': self.omega.copy(),
            
            
            'clock_time': np.array(time.time(), dtype=np.float32),
            # 'clock_time': np.array([time.time()]),
            'tau': self.tau_est.copy(),
            'cmd': self.cmd.copy(),
            'q_real': self.q_real.copy(),
            'dq_real': self.dq_real.copy(),
            'ddq_real': self.ddq_real.copy(),
            'accelerometer': np.array(self.low_state.imu_state.accelerometer).copy(),
            
        }
    _get_motion_to_save = _get_motion_to_save_np
            
    _motion_ref_pid = 1
    def LowStateCallback(self):
        """
            Basic Operation:
                - B: low level Quick Stop
                - X: high level Reset
            
            Policy Switching:
                - Left, Right: switch a motion Pid buffer, don't take effect immediately
                    - it's a round switch, 1->2->3->1
                - Down: go back to default policy
                - Up: go back to motion policy i.e. the motion policy Switched by the Left and Right
            
            
        """
        # handle joystick keyboard
        LowLevelMagic.LowStateCallback(self)
            
        
        if self.joystick.button[KeyMap.X] == 1: # quick reset
            self._ref_pid = -2
            
        if self._pid_size > 1:
            if self.joystick.button[KeyMap.down] == 1: # go back to default policy 
                self._ref_pid = 0
            
        
            if self.joystick.button[KeyMap.up] == 1: # go to motion policy 
                self._ref_pid = self._motion_ref_pid
                if not (self._ref_pid >= 1 and self._ref_pid < self._pid_size):
                    logger.error(f"Invalid policy id: {self._ref_pid} | pid_size: {self._pid_size}")
                    logger.info(f"Switching to the default policy")
                    self._ref_pid = 0
                            
            if self._tick % 100 == 1:
                if self.joystick.button[KeyMap.left] == 1: 
                    # self._motion_ref_pid -=1 
                    self._motion_ref_pid = (self._motion_ref_pid-2) % (self._pid_size-1) +1
                    logger.info(f"motion pid: switch to {self._motion_ref_pid}")
                    
                if self.joystick.button[KeyMap.right] == 1:  
                    # self._motion_ref_pid +=1 
                    self._motion_ref_pid = (self._motion_ref_pid) % (self._pid_size-1) +1
                    logger.info(f"motion pid: switch to {self._motion_ref_pid}")
                    # INFO:  self._motion_ref_pid should in the range of [1, len(cfg_policies)-1]
                        # it start from 1 because 0 should be the default locomotion policy
            print("\t\t\t\t\tReal Robot: _ref_pid: ", self._ref_pid, "| _motion_ref_pid", self._motion_ref_pid, end='\r\b')
        
        pass
            
    # self.low_cmd.motor_cmd is designed to be **Stateless**
    # Each time you call the following CMD function,
    # you should assume that the previous low_cmd is randomly generated
    def SetPose(self, pose: np.ndarray):
        assert pose.shape == (self.num_dofs,)
        
        for i, j in enumerate(self.dof_idx_23_to_29):
            self.low_cmd.motor_cmd[j].q = pose[i]
            self.low_cmd.motor_cmd[j].qd = 0
            self.low_cmd.motor_cmd[j].tau = 0
            # self.low_cmd.motor_cmd[j].kp = 0  # 使用预计算值
            # self.low_cmd.motor_cmd[j].kd = 0  # 使用预计算值
            self.low_cmd.motor_cmd[j].kp = self.kp_real[j]  # 使用预计算值
            self.low_cmd.motor_cmd[j].kd = self.kd_real[j]  # 使用预计算值
            
        # 锁定关节的kp/kd
        for j in self.dof_idx_locked:
            self.low_cmd.motor_cmd[j].q = 0
            self.low_cmd.motor_cmd[j].qd = 0
            self.low_cmd.motor_cmd[j].tau = 0
            # self.low_cmd.motor_cmd[j].kp = 0 # 使用预计算值
            # self.low_cmd.motor_cmd[j].kd = 0  # 使用预计算值
            self.low_cmd.motor_cmd[j].kp = self.kp_real[j]  # 使用预计算值
            self.low_cmd.motor_cmd[j].kd = self.kd_real[j]  # 使用预计算值
            
        self.send_cmd()
    
    def ToDamping(self):
        logger.info("Enter damping state.")
        logger.info("Waiting for the start signal...")
        while self.joystick.button[KeyMap.start] != 1:
            create_damping_cmd(self.low_cmd)
            self.send_cmd()
            time.sleep(self.dt)
    
    def ToZeroTorque(self):
        logger.info("Enter zero torque state.")
        logger.info("Waiting for the start signal...")
        while self.joystick.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd()
            time.sleep(self.dt)
    
    MoveToDefaultPose = lambda self: self.MoveToPose29Dof(self.dof_init_pose_real, 2.0)
    KeepDefaultPose = lambda self: self.KeepPose29Dof(self.dof_init_pose_real)

    def MoveToPose29Dof(self, target_pose: np.ndarray, duration: float):
        logger.info("Moving to specified pose.")
        num_step = int(duration / self.dt)
        
        self.GetState()
        start_dof_pos = self.q_real.copy()
        end_dof_pos = target_pose.copy()  # Use the provided target pose
        
        # Move to the specified pose
        for i in range(num_step + 1):
            alpha = i / num_step
            for j in range(self.num_real_dofs):
                self.low_cmd.motor_cmd[j].q = start_dof_pos[j] * (1 - alpha) + end_dof_pos[j] * alpha
                self.low_cmd.motor_cmd[j].qd = 0
                self.low_cmd.motor_cmd[j].kp = self.kp_real[j]  # Use pre-calculated values
                self.low_cmd.motor_cmd[j].kd = self.kd_real[j]  # Use pre-calculated values
                self.low_cmd.motor_cmd[j].tau = 0
            self.send_cmd()
            time.sleep(self.dt)

    def KeepPose29Dof(self, pose: np.ndarray):
        logger.info("Entering pose hold state.")
        logger.info("Waiting for the Button A signal...")
        
        # Maintain the current pose
        current_pose = pose.copy()
        
        while self.joystick.button[KeyMap.A] != 1:
            for j in range(self.num_real_dofs):
                self.low_cmd.motor_cmd[j].q = current_pose[j]
                self.low_cmd.motor_cmd[j].qd = 0
                self.low_cmd.motor_cmd[j].kp = self.kp_real[j]  # Use pre-calculated values
                self.low_cmd.motor_cmd[j].kd = self.kd_real[j]  # Use pre-calculated values
                self.low_cmd.motor_cmd[j].tau = 0
            self.send_cmd()
            time.sleep(self.dt)







