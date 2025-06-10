# Unified Robot Control Interface for HumanoidVerse
# Weiji Xie @ 2025.03.04

import mujoco, mujoco_viewer, mujoco.viewer
import glfw 
import os
import time
import signal
import sys
from pathlib import Path

import torch
from humanoidverse.deploy import URCIRobot
from scipy.spatial.transform import Rotation as R
import logging
from utils.config_utils import *  # noqa: E402, F403
# add argparse arguments

from typing import Dict, Optional, Tuple
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger
from humanoidverse.utils.helpers import np2torch, torch2np
from humanoidverse.utils.real.rotation_helper import get_gravity_orientation, transform_imu_data, quaternion_to_euler_array
from humanoidverse.data.robots.dtype import RobotExitException

import numpy as np

def wrap_to_pi_float(angles:float):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles



class MViewerPlugin:
    # TODO: visualize the motion keypoint in MujocoViewer
    is_pause = False
    
    def _make_viewer(self):
        def _key_callback(key):
                #  Keyboard mapping:
                #  ----------------
                #  |   K L ; '    |
                #  |   , . /      |
                #  ----------------
                key_char = chr(key)
                
                print(key, key_char)
                
                if key == 27: # Esc
                    self.viewer.close()
                
                if key_char == ' ':
                    self.is_pause = not self.is_pause
                
                if key_char == 'w':
                    self.cmd[1] += 0.1
                elif key_char == 's':
                    self.cmd[1] -= 0.1
                elif key_char == 'a': #
                    self.cmd[0] += 0.1
                elif key_char == 'd':
                    self.cmd[0] -= 0.1
                elif key_char == 'e':
                    if self.heading_cmd:
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]+np.pi/20)
                    else:
                        self.cmd[2] += 0.1
                elif key_char == 'q': 
                    if self.heading_cmd:                            
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]-np.pi/20)
                    else:
                        self.cmd[2] -= 0.1
                elif key_char == 'r':
                    self.cmd = np.array(self.cfg.deploy.defcmd)
                elif key == 257: # Enter
                    # self.Reset()
                    self._ref_pid = -2
                elif key_char == '[':
                    self._ref_pid -= 1
                elif key_char == ']':
                    self._ref_pid += 1
                
                print(self.cmd, self._ref_pid)
        # glfw.set_key_callback(self.viewer.window, _key_callback)
        ...
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=_key_callback)
        self.viewer.cam.lookat[:] = np.array([0,0,0.8])
        self.viewer.cam.distance = 3.0        
        self.viewer.cam.azimuth = 30                         # 可根据需要调整角度
        self.viewer.cam.elevation = -30                      # 负值表示从上往下看
        
        
    def render_step(self):
        while self.is_pause:
            time.sleep(0.01)
        
        if self.is_render:
            if self.viewer.is_running():
                self.viewer.sync()
            
                # breakpoint()
            else:
                raise RobotExitException("Mujoco Robot Exit")
        ...


class ViewerPlugin:
    # TODO: visualize the motion keypoint in MujocoViewer
    
    is_recording = False
    fps = 30
    
    def _make_viewer(self):
        ...
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.lookat[:] = np.array([0,0,0.6])
        self.viewer.cam.distance = 3.0        
        self.viewer.cam.azimuth = 180                         # 可根据需要调整角度
        self.viewer.cam.elevation = -20                      # 负值表示从上往下看
        def _key_callback(window, key, scancode, action, mods):
            if action == glfw.PRESS:
                #  Keyboard mapping:
                #  ----------------
                #  |   K L ; '    |
                #  |   , . /      |
                #  ----------------

                if key == glfw.KEY_COMMA:
                    self.cmd[1] += 0.1
                elif key == glfw.KEY_SLASH:
                    self.cmd[1] -= 0.1
                elif key == glfw.KEY_L: #
                    self.cmd[0] += 0.1
                elif key == glfw.KEY_PERIOD:
                    self.cmd[0] -= 0.1
                elif key == glfw.KEY_K:
                    if self.heading_cmd:
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]+np.pi/10)
                    else:
                        self.cmd[2] += 0.1
                elif key == glfw.KEY_SEMICOLON:
                    if self.heading_cmd:                            
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]-np.pi/10)
                    else:
                        self.cmd[2] -= 0.1
                elif key == glfw.KEY_APOSTROPHE:
                    self.cmd = np.array(self.cfg.deploy.defcmd)
                elif key == glfw.KEY_ENTER:
                    # self.Reset()
                    self._ref_pid = -2
                elif key == glfw.KEY_LEFT_BRACKET:
                    self._ref_pid -= 1
                elif key == glfw.KEY_RIGHT_BRACKET:
                    self._ref_pid += 1
                elif key>=320 and key<=329:
                    # 小键盘的数字 0-9 ,控制 策略切换
                    self._ref_pid = key-320
                elif key >= 48 and key <= 57:
                    # 0-9 ,控制 策略切换
                    self._ref_pid = key-48
                else:
                    print('Press key: ',key)
                
                print(self.cmd, self._ref_pid)
            self.viewer._key_callback(window, key, scancode, action, mods)
        glfw.set_key_callback(self.viewer.window, _key_callback)
        
        if self.is_recording:
            logger.info("Recording is True")
            self.start_time = time.time()
            self._video_buffer = []
            viewport = self.viewer.viewport
            self._frame_buffer = np.zeros((viewport.height, viewport.width, 3), dtype=np.uint8)
            self._frame_size = (viewport.height // self.macro_block_size * self.macro_block_size,  #
                                viewport.width // self.macro_block_size * self.macro_block_size)
    macro_block_size = 16
    def render_step(self):
        if self.is_render:
            # if self.viewer.is_running():
            if self.viewer.is_alive:
                self.viewer.render()
                # self.viewer.sync()
                if self.is_recording and (time.time() - self.start_time >= 1/self.fps):
                    self.start_time = time.time()
                    mujoco.mjr_readPixels(self._frame_buffer, None, self.viewer.viewport, self.viewer.ctx)
                    # imageio.imwrite('frame.png', frame[::-1])
                    # breakpoint()
                    
                    self._video_buffer.append(self._frame_buffer[::-1].copy()[:self._frame_size[0], :self._frame_size[1]])
            else:
                if self.is_recording:
                    logger.info("Mujoco: Saving video ...")
                    
                    model_path:Path = self.cfg.checkpoint
                    model_id = model_path.stem.replace('model_', 'ckpt_')
                    video_path:Path = model_path.parent.parent / 'renderings' / model_id / f'video_{time.strftime("%Y%m%d_%H%M%S")}.mp4'
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    import imageio
                    imageio.mimsave(video_path, self._video_buffer, fps=self.fps, macro_block_size=self.macro_block_size)
                    logger.info(f"Video saved to {video_path}")
                raise RobotExitException("Mujoco Robot Exit")
        

class MujocoRobot(URCIRobot, ViewerPlugin):
    REAL=False
    
    HANG=False
    
    RAND_NOISE: bool = False
    RAND_DELAY: bool = False
    RAND_MASK : bool = False
    
    noise_ratio = 3e-2
    # delay_ratio = (4, 25) # unit: ms
    delay_ratio = (4, 20) # unit: ms
    mask_ratio = 0.7
    mk_rand_noise = lambda tens, ratio: (
                                            (tens * (1 + torch.randn_like(tens) * ratio) ) 
                                    if isinstance(tens, torch.Tensor) 
                                    else    (tens * (1 + np.random.randn(*tens.shape).astype(tens.dtype) * ratio)) )# type: ignore
    # mk_rand_noise = lambda tens, ratio: tens * (1 + np.random.randn(*tens.shape).astype(tens.dtype) * ratio) # type: ignore
    
    # print_torque = lambda tau: print(f"tau (norm, max) = {torch.norm(tau):.2f}, \t{torch.max(tau):.2f}", end='\r')
    print_torque = lambda tau: print(f"tau (norm, max) = {np.linalg.norm(tau):.2f}, \t{np.max(tau):.2f}", end='\r')
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        def signal_handler(sig, frame):
            logger.info("Ctrl+C  Exiting safely...")
            raise RobotExitException("Mujoco Robot Exiting")
        signal.signal(signal.SIGINT, signal_handler)
        
        self.decimation = cfg.simulator.config.sim.control_decimation
        self.sim_dt = 1/cfg.simulator.config.sim.fps
        assert self.dt == self.decimation * self.sim_dt
        # self._subtimer = 0
        
        
        
        self.model = mujoco.MjModel.from_xml_path(os.path.join(cfg.robot.asset.asset_root, cfg.robot.asset.xml_file)) # type: ignore
        # self.model = mujoco.MjModel.from_xml_path('/home/bai/ASAP/humanoidverse/data/robots/g1/g1_23dof_lock_wrist_phys.xml')
        self.data = mujoco.MjData(self.model) # type: ignore
        self.model.opt.timestep = self.sim_dt
        if cfg.deploy.render:
            self.is_render = True
            if cfg.__contains__('recording') and cfg.recording:
                self.is_recording = True
            self._make_viewer()
        else:
            self.is_render = False
            
        self.num_ctrl = self.data.ctrl.shape[0]
        assert self.num_ctrl == self.num_actions, f"Number of control DOFs {self.num_ctrl} does not match number of actions {self.num_actions}"
        
        logger.info("Initializing Mujoco Robot")
        logger.info("Task Name: {}".format(cfg.log_task_name))
        logger.info("Robot Type: {}".format(cfg.robot.asset.robot_type))
        
        
        logger.info(f"decimation: {self.decimation}, sim_dt: {self.sim_dt}, dt: {self.dt}")
        logger.info(f"xml_file: {cfg.robot.asset.xml_file}")
        # print(self.decimation, self.sim_dt, self.dt)
        self.Reset()
        
        mujoco.mj_step(self.model, self.data) # type: ignore    


    def _reset(self):
        self.data.qpos[:3] = np.array(self.cfg.robot.init_state.pos)
        self.data.qpos[3:7] = np.array(self.cfg.robot.init_state.rot)[[3,0,1,2]] # XYZW to WXYZ
        self.data.qpos[7:] = self.dof_init_pose
        self.data.qvel[:]   = 0
        self.cmd = np.array(self.cfg.deploy.defcmd)
        
        #     self.data.qpos[7:]=np.array([-2.5090545e-01, -2.5106151e-02,  7.1244887e-03,  1.6603637e-01,
        #    -6.8755798e-02,  1.0320849e-02, -1.4015448e-01,  4.8405614e-02,
        #     2.2203762e-02,  1.5147208e-01, -1.6629830e-01, -5.5523389e-03,
        #     5.0889206e-04,  1.8234365e-03,  1.1872462e-02,  1.8569557e-01,
        #     1.7495771e-01, -4.6858322e-03,  9.0659463e-01,  1.8918298e-01,
        #    -1.7903234e-01,  1.5699334e-02,  9.0766126e-01], dtype=np.float32)
        
        
        #     self.data.qpos[7:] = np.array([-5.0120752e-02,  2.0163324e-02, -7.5798128e-03,  3.4919369e-01,
        #    -2.8446102e-01, -2.5288673e-02, -4.7107577e-02,  1.8228805e-02,
        #    -8.1556644e-03,  3.6581179e-01, -3.1244135e-01, -1.0373312e-02,
        #     3.3479740e-04, -2.2920404e-04, -1.2685804e-02,  2.0712337e-01,
        #     1.8589930e-01, -3.1398670e-03,  9.0953082e-01,  2.0744693e-01,
        #    -1.8387397e-01,  2.2290659e-03,  9.0941095e-01], dtype=np.float32)
        
    #     self.data.qpos[3:7] = np.array([ 0.6405637 , -0.00132383, -0.00757721, -0.7678665 ], dtype=np.float32)
    #     
    #     self.data.qpos[7:]=np.array([-5.96156046e-02, -2.54555605e-02,  9.49485414e-03,  3.70618284e-01,
    #     -2.84024119e-01,  1.78716499e-02, -4.85672913e-02, -1.99843608e-02,
    #     1.40748825e-02,  3.53395075e-01, -2.75969267e-01,  1.50737213e-02,
    #    -6.69594810e-05, -2.60704872e-03, -9.67030507e-03,  2.06643999e-01,
    #     1.82651579e-01, -1.89350755e-03,  9.21970427e-01,  2.05757156e-01,
    #    -1.83993816e-01,  9.43158567e-03,  9.09806430e-01], dtype=np.float32)
    
    #     self.data.qpos[3:7] = np.array([ 0.9177167 ,  0.06428247,  0.09726571, -0.3797411 ], dtype=np.float32)
    #     self.data.qpos[7:] = np.array([-2.5090545e-01, -2.5106151e-02,  7.1244887e-03,  1.6603637e-01,
    #    -6.8755798e-02,  1.0320849e-02, -1.4015448e-01,  4.8405614e-02,
    #     2.2203762e-02,  1.5147208e-01, -1.6629830e-01, -5.5523389e-03,
    #     5.0889206e-04,  1.8234365e-03,  1.1872462e-02,  1.8569557e-01,
    #     1.7495771e-01, -4.6858322e-03,  9.0659463e-01,  1.8918298e-01,
    #    -1.7903234e-01,  1.5699334e-02,  9.0766126e-01], dtype=np.float32)
        # breakpoint()
    #     self.data.qpos[3:] = np.array([9.9985272e-01,
    #     1.3556309e-02,  5.9258053e-03,  8.6955857e-03, -7.0489824e-02,
    #     1.3976265e-03,  4.9416097e-03,  3.5255992e-01, -2.6924220e-01,
    #    -1.6139226e-02, -3.3434451e-02, -5.3774547e-03,  8.0485297e-03,
    #     3.6053661e-01, -3.1571677e-01, -2.2485009e-03, -5.4906774e-04,
    #    -6.4945215e-04, -1.3742541e-02,  2.0349215e-01,  1.8248379e-01,
    #    -3.2597093e-03,  9.1346163e-01,  2.0483437e-01, -1.8628280e-01,
    #     3.4274885e-03,  9.2512226e-01], dtype=np.float32)
        

    @staticmethod
    def pd_control(target_q, q, kp, target_dq, dq, kd):
        '''Calculates torques from position commands
        '''
        if MujocoRobot.RAND_NOISE:
            kp,kd = MujocoRobot.mk_rand_noise(np.array([kp, kd]), MujocoRobot.noise_ratio)
        return (target_q - q) * kp + (target_dq - dq) * kd

    def GetState(self):
        '''Extracts physical states from the mujoco data structure
        '''
        data = self.data
        self.q = data.qpos.astype(np.double)[7:] # 19 dim
            # WXYZ
            # 3 dim base pos + 4 dim quat + 12 dim actuation angles
        self.dq = data.qvel.astype(np.double)[6:] # 18 dim ?????
            # 3 dim base vel + 3 dim omega + 12 dim actuation vel
        
        self.pos = data.qpos.astype(np.double)[:3]
        self.quat = data.qpos.astype(np.double)[3:7][[1,2,3,0]] # WXYZ to XYZW
        self.vel = data.qvel.astype(np.double)[:3]
        self.omega = data.qvel.astype(np.double)[3:6]
        
        r = R.from_quat(self.quat)  # R.from_quat: need xyzw
        self.rpy = quaternion_to_euler_array(self.quat) # need xyzw
        self.rpy[self.rpy > math.pi] -= 2 * math.pi
        self.gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    
        
        if self.heading_cmd:
            self.cmd[2] = np.clip(0.5*wrap_to_pi_float(self.cmd[3] - self.rpy[2]), -1., 1.)
            
        if self.is_motion_tracking:
            self.motion_time = (self.timer) * self.dt 
            self.ref_motion_phase = self.motion_time / self.motion_len
            
        # breakpoint()
    
    def _sanity_check(self, target_q):
        unsafe_dof = np.where((np.abs(target_q - self.q) > 2.2 ) | 
                              (np.abs(self.dq) > 20))[0]
        if len(unsafe_dof) > 0:
            for motor_idx in unsafe_dof:
                logger.error(f"Action of joint {motor_idx} is too large.\n"
                                f"target q\t: {target_q[motor_idx]} \n"
                                f"target dq\t: {0} \n"
                                f"q\t\t: {self.q[motor_idx]} \n"
                                f"dq\t\t: {self.dq[motor_idx]}\n")
                # breakpoint()  
        
        
    def _apply_action(self, target_q):
        
        self._sanity_check(target_q)
        
        rand_mask = np.random.random(self.num_actions) < self.mask_ratio

        rand_delay = np.random.randint((1e-3*self.delay_ratio[0])//self.sim_dt, (1e-3*self.delay_ratio[1])//self.sim_dt) * self.RAND_DELAY
        step_delay = rand_delay//self.decimation
        substep_delay = rand_delay - step_delay * self.decimation
        
        if step_delay ==0:
            old_action = self.history_handler.history['actions'][0, step_delay]
            old_trg_q = np.clip(old_action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
            cur_trg_q = target_q
        else:
            old_action = self.history_handler.history['actions'][0, step_delay]
            cur_action = self.history_handler.history['actions'][0, step_delay+1]
            old_trg_q = np.clip(old_action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
            cur_trg_q = np.clip(cur_action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
                
        for i in range(self.decimation):
            self.GetState()
            
            if self.RAND_DELAY and i < substep_delay:
                target_q = old_trg_q
            elif self.RAND_MASK:
                target_q = cur_trg_q * rand_mask + old_trg_q.numpy() * (1 - rand_mask)
            else:
                target_q = cur_trg_q
                
            tau = self.pd_control(target_q, self.q, self.kp,
                            0, self.dq, self.kd)  # Calc torques
            
            if self.RAND_NOISE: tau = MujocoRobot.mk_rand_noise(tau, MujocoRobot.noise_ratio)
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)  # Clamp torques
            
            
            # MujocoRobot.print_torque(tau)
            # tau*=0
            # print(np.linalg.norm(target_q-self.q), np.linalg.norm(self.dq), np.linalg.norm(tau))
            # self.data.qpos[:3] = np.array([0,0,1])
            self.data.ctrl[:] = tau
            if self.HANG:
                # self.data.ctrl[14] = 0.5
                self.data.qpos[:3] = np.array([0,0,1])

            mujoco.mj_step(self.model, self.data) # type: ignore
            
            self.tracking()
            self.render_step()
            # self._subtimer += 1
        
        
    # Noise Version ApplyAction & Obs
    def ApplyAction(self, action): 
        if self.RAND_NOISE: action = MujocoRobot.mk_rand_noise(action, MujocoRobot.noise_ratio)
        URCIRobot.ApplyAction(self, action)


            
        # breakpoint()

    def Obs(self):
        
        # return {k: torch2np(v) for k, v in self.obs_buf_dict.items()}
        
        actor_obs = torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)
        if self.RAND_NOISE:
            actor_obs = MujocoRobot.mk_rand_noise(actor_obs, MujocoRobot.noise_ratio)
        return {
            'actor_obs': actor_obs
            }
        

    def _get_motion_to_save_torch(self)->Tuple[float, Dict[str, torch.Tensor]]:
        
        from scipy.spatial.transform import Rotation as sRot
        
        motion_time = (self.timer) * self.dt 

        root_trans = self.pos
        root_rot = self.quat # XYZW
        root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot).as_rotvec()).float() # type: ignore
        dof = self.q
        # T, num_env, J, 3
        # print(self._motion_lib.mesh_parsers.dof_axis)
        pose_aa = torch.cat([root_rot_vec[..., None, :], 
                                torch.from_numpy(self._dof_axis * dof[..., None]), 
                                torch.zeros((self.num_augment_joint, 3))], axis = 0) # type: ignore
        
        return motion_time, {
            'root_trans_offset': torch.from_numpy(root_trans),
            'root_rot': torch.from_numpy(root_rot), # 统一save xyzw
            'dof': torch.from_numpy(dof),
            'pose_aa': pose_aa,
            'action': torch.from_numpy(self.act),
            'actor_obs': np2torch(self.obs_buf_dict['actor_obs']),
            'terminate': torch.zeros((1,)),
            'dof_vel': torch.from_numpy(self.dq),
            'root_lin_vel': torch.from_numpy(self.vel),
            'root_ang_vel': torch.from_numpy(self.omega),
            
            'clock_time': torch.from_numpy(np.array([time.time()])),
            'tau': torch.from_numpy(self.data.ctrl).clone(),
            'cmd': torch.from_numpy(self.cmd)
        }
        
    def _get_motion_to_save_np(self)->Tuple[float, Dict[str, np.ndarray]]:
        
        from scipy.spatial.transform import Rotation as sRot
        
        motion_time = (self.timer) * self.dt 

        root_trans = self.pos
        root_rot = self.quat # XYZW
        root_rot_vec = np.array(sRot.from_quat(root_rot).as_rotvec(), dtype=np.float32) # type: ignore
        dof = self.q
        # T, num_env, J, 3
        # print(self._motion_lib.mesh_parsers.dof_axis)
        pose_aa = np.concatenate([root_rot_vec[..., None, :], 
                                np.array(self._dof_axis * dof[..., None]), 
                                np.zeros((self.num_augment_joint, 3))], axis = 0) # type: ignore
        
        return motion_time, {
            'root_trans_offset': (root_trans).copy(),
            'root_rot': (root_rot).copy(), # 统一save xyzw
            'dof': (dof).copy(),
            'pose_aa': pose_aa,
            'action': (self.act).copy(),
            'actor_obs': (self.obs_buf_dict['actor_obs']),
            'terminate': np.zeros((1,)),
            'dof_vel': (self.dq).copy(),
            'root_lin_vel': (self.vel).copy(),
            'root_ang_vel': (self.omega).copy(),
            
            'clock_time': (np.array([time.time()])),
            'tau': (self.data.ctrl).copy(),
            'cmd': (self.cmd).copy()
        }
        
    _get_motion_to_save = _get_motion_to_save_np


    def tracking(self):
        # breakpoint()
        # print(np.linalg.norm(self.data.xpos[6]-self.data.xpos[12]))
        if np.any(self.data.contact.pos[:,2] > 0.01):
            names_list = self.model.names.decode('utf-8').split('\x00')[:40]
            res = np.zeros((6,1),dtype=np.float64)
            geom_name = lambda x: (names_list[self.model.geom_bodyid[x] + 1])
            geom_force = lambda x:mujoco.mj_contactForce(self.model,self.data,x,res) #type:ignore
            
            for contact in self.data.contact:
                if contact.pos[2] > 0.01 and contact.geom1 != 0 and contact.geom2 != 0:
                    geom1_name = geom_name(contact.geom1)
                    geom2_name = geom_name(contact.geom2)
                    logger.warning(f"Warning!!! Collision between '{geom1_name,contact.geom1}' and '{geom2_name,contact.geom2}' at position {contact.pos}.")
                    # breakpoint()
                    
