# Unified Robot Control Interface for HumanoidVerse
# Weiji Xie @ 2025.03.04

import mujoco, mujoco_viewer, mujoco.viewer
import glfw 
import os
import time
import signal
import sys
from pathlib import Path
import joblib
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
from io import BytesIO
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree

from isaac_utils.rotations import (
    my_quat_rotate,
    calc_heading_quat_inv,
    calc_heading_quat,
    quat_mul,
    quat_conjugate,
    quat_to_angle_axis,
    xyzw_to_wxyz,
    wxyz_to_xyzw
)
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
    HOI_interaction: bool = True
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
        
        
        if self.HOI_interaction:
            self.xml_path = os.path.join(cfg.robot.asset.asset_root, cfg.robot.asset.xml_file)
            with open(self.xml_path, "rb") as f:
                robot_xml_data = f.read()
            data_seq = joblib.load(cfg.robot.motion.motion_file)
            #self.obj_name = list(data_seq.keys())
            self.obj_name = list(["largebox"])#list(data_seq.keys())
            self.TASK_ID = 0
            self._motion_lib.load_motions(random_sample=False)
            self.motion_length = self._motion_lib.get_motion_length()[self.TASK_ID]
            motion_res = self._motion_lib.get_motion_state([self.TASK_ID], torch.tensor(0))
            init_obj_pose = motion_res['o_rb_pos'][0]
            new_string = self.xml_add_obj(robot_xml_data,self.obj_name[self.TASK_ID], init_obj_pose)
            xml_with_object = f"humanoidverse/data/robots/g1/g1_29dof_with_hand_obj.xml"
            open(xml_with_object, "wb").write(new_string)
            self.model = mujoco.MjModel.from_xml_path(xml_with_object)
        else:
            self.model = mujoco.MjModel.from_xml_path(os.path.join(cfg.robot.asset.asset_root, cfg.robot.asset.xml_file)) # type: ignore
        
        self.data = mujoco.MjData(self.model) # type: ignore
        self.model.opt.timestep = self.sim_dt
        self.dof_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.actuator_trnid[i, 0]) for i in range(self.model.nu)]
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
  
    def xml_add_obj(self, xml, obj_name, init_obj_pose):
        parser = XMLParser(remove_blank_text=True)
        tree = parse(BytesIO(xml), parser=parser)
        worldroot = tree.getroot().find("worldbody")
        obj_node = Element("body", {"name": obj_name, "pos": f"{init_obj_pose[0]} {init_obj_pose[1]} {init_obj_pose[2]}"})
        SubElement(
            obj_node,
            "joint",
            {"limited": "false", "name": obj_name, "type": "free"},
        )
        SubElement(
            obj_node,
            "geom",
            {
                "type": "mesh",
                "mesh": obj_name,
                "contype": "1",
                "conaffinity": "1",
                "friction": "0.8 0.01 0.8",
                "density": "25",
                "rgba": "1. 1. 1. 1",
            },
        )
        worldroot.append(obj_node)
        asset = tree.getroot().find("asset")
        asset.append(
            Element(
                "mesh",
                {
                    "file": f"../objects/objects/{obj_name}/{obj_name}.stl",
                },
            )
        )
        return etree.tostring(tree)

    def _reset(self):
        motion_times = torch.tensor((self.timer) * self.dt) # next frames so +1 
        motion_res = self._motion_lib.get_motion_state([0], motion_times, offset=None)
        root_pos = motion_res['root_pos'][0]
        root_rot = motion_res['root_rot'][0]
        root_vel = motion_res['root_vel'][0]
        root_ang_vel = motion_res['root_ang_vel'][0]

        dof_pos = motion_res['dof_pos'][0]
        dof_vel = motion_res['dof_vel'][0]
        # init robot
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = xyzw_to_wxyz(root_rot)
        self.data.qpos[7:7 + self.num_actions] = dof_pos  
        self.data.qvel[6:6 + self.num_actions] = dof_vel
        self.data.qvel[0:3] = root_vel 
        self.data.qvel[3:6] = root_ang_vel
        self.cmd = np.array(self.cfg.deploy.defcmd)

        if self.HOI_interaction:
            ref_o_ang_vel = motion_res['o_ang_vel'][0]
            ref_o_lin_vel = motion_res['o_lin_vel'][0]
            ref_o_rb_rot = motion_res['o_rb_rot'][0]
            ref_o_rb_pos = motion_res['o_rb_pos'][0]
            self.data.qpos[7+ self.num_actions : 7+ self.num_actions +3] = ref_o_rb_pos
            self.data.qpos[7+ self.num_actions + 3: 7+ self.num_actions +7] = xyzw_to_wxyz(ref_o_rb_rot) # TODO: simulator type protect
            self.data.qvel[6 + self.num_actions:6 + self.num_actions + 3] = ref_o_lin_vel
            self.data.qvel[6 + self.num_actions + 3:6 + self.num_actions + 6] = ref_o_ang_vel
        print(self.data.qpos[:3], self.data.qpos[7+ self.num_actions : 7+ self.num_actions +3])
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
        self.q = data.qpos.astype(np.double)[7:7+ self.num_actions] # 19 dim # 43dim
        self.dq = data.qvel.astype(np.double)[6:6+ self.num_actions] # 18 dim # 43dim

        if self.HOI_interaction:
            self._obs_global_obj_pos = data.qpos.astype(np.double)[7+ self.num_actions : 7+ self.num_actions +3]
            self._obs_global_obj_vel = data.qvel.astype(np.double)[6+ self.num_actions : 6+ self.num_actions +3]
            self.global_obj_rot = data.qpos.astype(np.double)[7+ self.num_actions + 3: 7+ self.num_actions +7][[1,2,3,0]] # WXYZ to XYZW
        motion_times = torch.tensor((self.timer) * self.dt) # next frames so +1
        motion_res = self._motion_lib.get_motion_state([0], motion_times)

        # ref_obj_pos = motion_res["o_rb_pos"][0] # TODO: add table pos
        # ref_obj_rot = motion_res["o_rb_rot"][0]
        # ref_obj_lin = motion_res["o_lin_vel"][0]
        # if "use motion obj obs":
        #     self._obs_global_obj_pos = ref_obj_pos.to(torch.float64)
        #     self.global_obj_rot = ref_obj_rot.to(torch.float64)
        #     self._obs_global_obj_vel = ref_obj_lin.to(torch.float64)
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
            
        # if self.is_motion_tracking:
        #     self.motion_time = (self.timer) * self.dt 
        #     self.ref_motion_phase = self.motion_time / self.motion_len

        if self.is_motion_tracking:
            self.motion_time = (self.timer) * self.dt
            self.ref_motion_phase = self.motion_time / self.motion_duration
            # self.ref_motion_phase = (motion_times / self.motion_length) #(Env,)
            # self.motions_phase[np.arange(self.num_envs), self.motion_indices] = self.ref_motion_phase + 1 # +1 for larger value, (Env, num_motions)
            self.motions_phase = [self.ref_motion_phase + 1, 0.,0.,0.] # +1 for larger value, (Env, num_motions)
    
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
        
        ##### TODO! #####
        # rand_mask = np.random.random(self.num_actions) < self.mask_ratio

        # rand_delay = np.random.randint((1e-3*self.delay_ratio[0])//self.sim_dt, (1e-3*self.delay_ratio[1])//self.sim_dt) * self.RAND_DELAY
        # step_delay = rand_delay//self.decimation
        # substep_delay = rand_delay - step_delay * self.decimation
        
        # if step_delay ==0:
        #     old_action = self.history_handler.history['actions'][0, step_delay]
        #     old_trg_q = np.clip(old_action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
        #     cur_trg_q = target_q
        # else:
        #     old_action = self.history_handler.history['actions'][0, step_delay]
        #     cur_action = self.history_handler.history['actions'][0, step_delay+1]
        #     old_trg_q = np.clip(old_action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
        #     cur_trg_q = np.clip(cur_action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
        ######        
        print(self.ref_motion_phase)
        for i in range(self.decimation):
            q = self.data.qpos[7:7+ self.num_actions]   
            dq = self.data.qvel[6:6+ self.num_actions]
            #### TODO ####
            # if self.RAND_DELAY and i < substep_delay:
            #     target_q = old_trg_q
            # elif self.RAND_MASK:
            #     target_q = cur_trg_q * rand_mask + old_trg_q.numpy() * (1 - rand_mask)
            # else:
            #     target_q = cur_trg_q
                
            tau = self.pd_control(target_q, q, self.kp, 0, dq, self.kd)  # Calc torques
            if self.RAND_NOISE: tau = MujocoRobot.mk_rand_noise(tau, MujocoRobot.noise_ratio)
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)  # Clamp torques
            
            self.data.ctrl[:] = tau
            if self.HANG:
                # self.data.ctrl[14] = 0.5
                self.data.qpos[:3] = np.array([0,0,1])

            mujoco.mj_step(self.model, self.data) # type: ignore
            
            #self.tracking()
            self.render_step()
            # self._subtimer += 1
        
        
    # Noise Version ApplyAction & Obs
    def ApplyAction(self, action): 
        if self.RAND_NOISE: action = MujocoRobot.mk_rand_noise(action, MujocoRobot.noise_ratio)
        URCIRobot.ApplyAction(self, action)


            
        # breakpoint()

    def Obs(self):
        #print(torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1))
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
            names_list = self.model.names.decode('utf-8').split('\x00')[:] #:40
            res = np.zeros((6,1),dtype=np.float64)
            geom_name = lambda x: (names_list[self.model.geom_bodyid[x]+1]) # 
            geom_force = lambda x:mujoco.mj_contactForce(self.model,self.data,x,res) #type:ignore
            
            for contact in self.data.contact:
                if contact.pos[2] > 0.01 and contact.geom1 != 0 and contact.geom2 != 0:
                    geom1_name = geom_name(contact.geom1)
                    geom2_name = geom_name(contact.geom2)
                    logger.warning(f"Warning!!! Collision between '{geom1_name,contact.geom1}' and '{geom2_name,contact.geom2}' at position {contact.pos}.")
                    # breakpoint()
                    
