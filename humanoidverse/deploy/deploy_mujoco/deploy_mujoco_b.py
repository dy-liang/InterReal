import os
import sys
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from humanoidverse.utils.logging import HydraLoggerBridge
from humanoidverse.utils.helpers import parse_observation
from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from humanoidverse.utils.motion_lib.hoi_motion_lib_base import MotionLibHoi
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot

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
import logging
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
import onnx
import onnxruntime
from io import BytesIO
from lxml.etree import XMLParser, parse, ElementTree, Element, SubElement
from lxml import etree

import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import joblib
# add argparse arguments
from loguru import logger
JIT_Policy = True
ONNX_Policy = False
class MujocoDeployer():
    def __init__(self, config):
        checkpoint_path = config.checkpoint
        self.config = config
        self.asset_root = config.robot.asset.asset_root
        self.xml_path = os.path.join(self.asset_root, config.robot.asset.xml_file)
        self.device = "cpu"
        self.load_obj_flag = True
        self.num_actions = config.robot.actions_dim 
        self.history_handler = HistoryHandler(1, config.obs.obs_auxiliary, config.obs.obs_dims, self.device)
        simulation_dt = 1.0 / config.simulator.config.sim.fps
        self.control_decimation = config.simulator.config.sim.control_decimation
        self.dt = simulation_dt * self.control_decimation

        # Load robot model and object
        if self.load_obj_flag:
            with open(self.xml_path, "rb") as f:
                robot_xml_data = f.read()
            data_seq = joblib.load(config.robot.motion.motion_file)
            self.obj_name = list(data_seq.keys())
            self._motion_lib = MotionLibHoi(config.robot.motion, num_envs=1, device=self.device)
            self._motion_lib.load_motions(random_sample=False)
            self.motion_length = self._motion_lib.get_motion_length()
            self.motion_start_times = 0
            motion_res = self._motion_lib.get_motion_state(0, torch.tensor(0))
            init_obj_pose = motion_res['o_rb_pos']
            new_string = self.xml_add_obj(robot_xml_data,self.obj_name[0], init_obj_pose)
            xml_with_object = f"humanoidverse/data/robots/g1/g1_29dof_with_hand_obj.xml"
            open(xml_with_object, "wb").write(new_string)
            self.mjmodel = mujoco.MjModel.from_xml_path(xml_with_object)
        else:
            self._motion_lib = MotionLibRobot(config.robot.motion, num_envs=1, device=self.device)
            self._motion_lib.load_motions(random_sample=False)
            self.motion_length = self._motion_lib.get_motion_length()
            self.motion_start_times = 0

            self.mjmodel = mujoco.MjModel.from_xml_path(self.xml_path)
        self.num_envs = 1
        self._init_motion_extend()
        self.mjdata = mujoco.MjData(self.mjmodel)
        self.mjmodel.opt.timestep = simulation_dt

        # load policy
        checkpoint_dir = os.path.dirname(checkpoint_path)
        exported_policy_path = os.path.join(checkpoint_dir, 'exported')
        exported_policy_name = checkpoint_path.split('/')[-1]
        if ONNX_Policy:
            exported_onnx_name = exported_policy_name.replace('.pt', '.onnx')
            onnx_path = os.path.join(exported_policy_path, exported_onnx_name)
            onnx_model = onnx.load(onnx_path)
            self.ort_session = onnxruntime.InferenceSession(onnx_path)
            self.input_name = self.ort_session.get_inputs()[0].name
        if JIT_Policy:
            pt_model_path = os.path.join(exported_policy_path, exported_policy_name)
            self.policy = torch.jit.load(pt_model_path)

        # load param
        self.load_pd_param(self.mjmodel, config)
        self.torque_limits = np.array(config.robot.dof_effort_limit_list)
        self.clip_action_limit = np.array(config.robot.control.action_clip_value)
        self.target_dof_pos = self.default_dof_pos.copy()
        self.episode_length_buf=0
        self.pause = False
        self.viewer = mujoco.viewer.launch_passive(self.mjmodel, self.mjdata, key_callback=self.key_call_back)

        # dof setting
        self.dof_names = [mujoco.mj_id2name(self.mjmodel, mujoco.mjtObj.mjOBJ_JOINT, self.mjmodel.actuator_trnid[i, 0]) for i in range(self.mjmodel.nu)]
        self.open_dof_indices = [self.dof_names.index(dof) for dof in self.config.robot.hybrid_control.hand_open_dof]
        self.open_dof_nums = len(self.open_dof_indices)
        self.rl_action_indices = [i for i in range(self.num_actions) if i not in self.open_dof_indices]
        self.rl_action_nums = len(self.rl_action_indices)
        self.motions_phase = torch.zeros(self.num_envs, len(self.obj_name), dtype=torch.float, device=self.device, requires_grad=False)
        self.motion_indices = torch.arange(self.num_envs) % 4

        self.reset(self.load_obj_flag)
        if ONNX_Policy:
            obs_tensor = self.obs_buf_dict["actor_obs"].unsqueeze(0).float().numpy()
            self.action = np.squeeze(self.ort_session.run(None, {self.input_name: obs_tensor})[0])
        if JIT_Policy:
            obs_tensor = self.obs_buf_dict["actor_obs"].unsqueeze(0).float()
            self.action = self.policy(obs_tensor)[0].detach().numpy()
        pass
    
    def load_pd_param(self, m, config):
        self.default_dof_pos = np.zeros(self.num_actions, dtype=np.float32)
        self.p_gains = np.zeros(self.num_actions, dtype=np.float32)
        self.d_gains = np.zeros(self.num_actions, dtype=np.float32)
        
        for i in range(self.num_actions): 
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i+1) # 0 is floating_base_joint
            angle = config.robot.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in config.robot.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = config.robot.control.stiffness[dof_name]
                    self.d_gains[i] = config.robot.control.damping[dof_name]
                    found = True
                    logger.debug(f"PD gain of joint {name} were defined, setting them to {self.p_gains[i-1]} and {self.d_gains[i-1]}")
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if config.robot.control.control_type in ["P", "V"]:
                    logger.warning(f"PD gain of joint {name} were not defined, setting them to zero")
                    raise ValueError(f"PD gain of joint {name} were not defined. Should be defined in the yaml file.")

        pass

    def _init_motion_extend(self):
        if "extend_config" in self.config.env.config.robot.motion:
            extend_parent_ids, extend_pos, extend_rot = [], [], []
            for extend_config in self.config.env.config.robot.motion.extend_config:
                extend_parent_ids.append(mujoco.mj_name2id(self.mjmodel, mujoco.mjtObj.mjOBJ_BODY, extend_config["parent_name"]))
                # extend_parent_ids.append(self.simulator.find_rigid_body_indice(extend_config["parent_name"]))
                extend_pos.append(extend_config["pos"])
                extend_rot.append(extend_config["rot"])
                # self.simulator._body_list.append(extend_config["joint_name"])

            self.extend_body_parent_ids = torch.tensor(extend_parent_ids, device=self.device, dtype=torch.long)
            self.extend_body_pos_in_parent = torch.tensor(extend_pos).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_wxyz = torch.tensor(extend_rot).repeat(self.num_envs, 1, 1).to(self.device)
            self.extend_body_rot_in_parent_xyzw = self.extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
            self.num_extend_bodies = len(extend_parent_ids)
            self.num_bodies = 44
            self.ref_body_pos_extend = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.dif_global_body_pos = torch.zeros(self.num_envs, self.num_bodies + self.num_extend_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)

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
                "friction": "0.9 0.01 0.0001",
                "mass": "0.5",
                # "stiffness": "30000",
                # "damping": "1000",
                "rgba": "1. 1. 1. 1",
                # "rgba": "0.8 0.6 .4 1",
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
    
    def reset(self, with_object=False):
        env_ids = 0
        motion_times = torch.tensor((self.episode_length_buf) * self.dt + self.motion_start_times) # next frames so +1
        motion_res = self._motion_lib.get_motion_state([0], motion_times)
        
        root_pos_noise = self.config.env.config.init_noise_scale.root_pos * self.config.env.config.noise_to_initial_level
        root_rot_noise = self.config.env.config.init_noise_scale.root_rot * 3.14 / 180 * self.config.env.config.noise_to_initial_level
        root_vel_noise = self.config.env.config.init_noise_scale.root_vel * self.config.env.config.noise_to_initial_level
        root_ang_vel_noise = self.config.env.config.init_noise_scale.root_ang_vel * self.config.env.config.noise_to_initial_level

        root_pos = motion_res['root_pos'][env_ids]
        root_rot = motion_res['root_rot'][env_ids]
        root_vel = motion_res['root_vel'][env_ids]
        root_ang_vel = motion_res['root_ang_vel'][env_ids]

        dof_pos = motion_res['dof_pos'][env_ids]
        dof_vel = motion_res['dof_vel'][env_ids]
        # init robot
        root_pos_randn = torch.randn_like(root_pos) * root_pos_noise
        self.mjdata.qpos[:3] = root_pos + root_pos_randn
        self.mjdata.qpos[3:7] = xyzw_to_wxyz(root_rot)
        self.mjdata.qpos[7:7 + len(self.target_dof_pos)] = dof_pos  # 设置关节位置
        self.mjdata.qvel[6:6 + len(self.target_dof_pos)] = dof_vel # 设置关节速度为 0
        self.mjdata.qvel[0:3] = root_vel + torch.randn_like(root_vel) * root_vel_noise
        self.mjdata.qvel[3:6] = root_ang_vel + torch.randn_like(root_ang_vel) * root_ang_vel_noise

        if with_object:
            ref_o_ang_vel = motion_res['o_ang_vel'][env_ids]
            ref_o_lin_vel = motion_res['o_lin_vel'][env_ids]
            ref_o_rb_rot = motion_res['o_rb_rot'][env_ids]
            ref_o_rb_pos = motion_res['o_rb_pos'][env_ids]
            self.mjdata.qpos[7+ self.num_actions : 7+ self.num_actions +3] = ref_o_rb_pos + root_pos_randn
            self.mjdata.qpos[7+ self.num_actions + 3: 7+ self.num_actions +7] = xyzw_to_wxyz(ref_o_rb_rot) # TODO: simulator type protect
            self.mjdata.qvel[6 + len(self.target_dof_pos):6 + len(self.target_dof_pos) + 3] = ref_o_lin_vel
            self.mjdata.qvel[6 + len(self.target_dof_pos) + 3:6 + len(self.target_dof_pos) + 6] = ref_o_ang_vel

        self.action = np.zeros(self.rl_action_nums, dtype=np.float32)
        
        self.step(self.action)
    
        
    def compute_observation(self):
        self.obs_buf_dict_raw = {}
        self.hist_obs_dict = {}
        
        for obs_key, obs_config in self.config.obs.obs_dict.items():
            if obs_key != "actor_obs":
                continue
            self.obs_buf_dict_raw[obs_key] = dict()
            parse_observation(self, obs_config, self.obs_buf_dict_raw[obs_key], self.config.obs.obs_scales, self.config.obs.noise_scales, 1.0)

        history_obs_list = self.history_handler.history.keys()
        parse_observation(self, history_obs_list, self.hist_obs_dict, self.config.obs.obs_scales, self.config.obs.noise_scales, 1.0)
        self.obs_buf_dict = dict()
        for obs_key, obs_config in self.config.obs.obs_dict.items():
            if obs_key != "actor_obs":
                continue
            obs_keys = sorted(obs_config)
            self.obs_buf_dict[obs_key] = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1)
            
        pass
    
    def run(self):            
        while self.viewer.is_running():
            if self.pause:
                time.sleep(0.01)
                continue
            if ONNX_Policy:
                obs_tensor = self.obs_buf_dict["actor_obs"].unsqueeze(0).float().numpy()
                self.action = np.squeeze(self.ort_session.run(None, {self.input_name: obs_tensor})[0])
            if JIT_Policy:
                obs_tensor = self.obs_buf_dict["actor_obs"].unsqueeze(0).float()
                self.action = self.policy(obs_tensor)[0].detach().numpy()
            self.action = np.clip(self.action, -self.clip_action_limit, self.clip_action_limit)
            actions_scaled = self.action * 0.25
            self.step(actions_scaled)
        
    def _pre_compute_observations_callback(self):
        motion_times=((self.episode_length_buf+1) * self.dt+self.motion_start_times)% self.motion_length
        quat = self.mjdata.qpos[3:7]
        env_batch_size = 1
        
        self.base_lin_vel = torch.from_numpy(self.mjdata.qvel[:3])
        self.base_ang_vel = torch.from_numpy(self.mjdata.qvel[3:6])
        self.projected_gravity = torch.from_numpy(self.get_gravity_orientation(quat))
        self.dof_pos = torch.from_numpy(self.mjdata.qpos[7:7+ self.num_actions])
        self.dof_vel = torch.from_numpy(self.mjdata.qvel[6:6+ self.num_actions])
        self.action_torch = torch.from_numpy(self.action)
        self._ref_motion_phase = torch.tensor([motion_times / self.motion_length])
        self._ref_motion_phase = (motion_times / self.motion_length) #(Env,)
        self.motions_phase = torch.zeros(self.num_envs, len(self.obj_name), dtype=torch.float, device=self.device, requires_grad=False)
        self.motion_indices = torch.arange(self.num_envs) % 4
        self.motions_phase[np.arange(self.num_envs), self.motion_indices] = self._ref_motion_phase + 1 # +1 for larger value, (Env, num_motions)
        root_pos = torch.tensor(self.mjdata.qpos[:3])
        
        rigid_pos = torch.from_numpy(self.mjdata.xpos).float() #(46,3)
        rigid_rot = torch.from_numpy(self.mjdata.xquat).float()
        rigid_body_pos = rigid_pos[1:-1]
        # rigid_body_rot = rigid_rot[1:-1]
        rigid_body_pos_24 = rigid_body_pos[[i for i in range(rigid_body_pos.shape[0]) if i not in [29,28,27,22,21,20]]]
        
        rotated_pos_in_parent = my_quat_rotate(
           wxyz_to_xyzw(rigid_rot[None, self.extend_body_parent_ids].reshape(-1, 4)),
            self.extend_body_pos_in_parent.reshape(-1, 3)
        )
        extend_curr_pos = my_quat_rotate(
            self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
            rotated_pos_in_parent
        ).view(self.num_envs, -1, 3) + rigid_pos[None, self.extend_body_parent_ids]
        self._rigid_body_pos_extend = torch.cat([rigid_body_pos_24[None], extend_curr_pos], dim=1)

        num_rigid_bodies = self._rigid_body_pos_extend.shape[1]
        motion_times = torch.tensor((self.episode_length_buf) * self.dt + self.motion_start_times) # next frames so +1
        motion_res = self._motion_lib.get_motion_state([0], motion_times)
        
        heading_inv_rot = calc_heading_quat_inv(wxyz_to_xyzw(torch.from_numpy(self.mjdata.qpos[3:7])).clone().unsqueeze(0), w_last=True)
        
        # ================== Object Interaction =====================
        if self.load_obj_flag:  # TODO: maybe read param tree will cast time
        
            ref_obj_pos = motion_res["o_rb_pos"] # TODO: add table pos
            ref_obj_rot = motion_res["o_rb_rot"]
            ref_obj_lin = motion_res["o_lin_vel"]
            ref_obj_ang = motion_res["o_ang_vel"]

            global_obj_pos = torch.from_numpy(self.mjdata.qpos[7+ self.num_actions : 7+ self.num_actions +3]).unsqueeze(0)
            self.global_obj_rot = wxyz_to_xyzw(torch.from_numpy(self.mjdata.qpos[7+ self.num_actions + 3: 7+ self.num_actions +7])).unsqueeze(0)
            global_obj_lin = torch.from_numpy(self.mjdata.qvel[6 + len(self.target_dof_pos):6 + len(self.target_dof_pos) + 3]).unsqueeze(0)
            global_obj_ang = torch.from_numpy(self.mjdata.qvel[6 + len(self.target_dof_pos) + 3:6 + len(self.target_dof_pos) + 6]).unsqueeze(0)
            #print("Current Obj Pos: ", global_obj_pos, self.mjdata.qpos[7+ self.num_actions + 3: 7+ self.num_actions +7], "Current Human Pos: ", root_pos, self.mjdata.qpos[3:7] )

            ## diff compute - kinematic object
            local_obj_root_pos = global_obj_pos.view(env_batch_size, -1, 3) - root_pos.view(env_batch_size, 1, 3)
            local_obj_pos_flat = my_quat_rotate(heading_inv_rot.view(-1, 4), local_obj_root_pos.view(-1, 3))
            self._obs_local_obj_pos = local_obj_pos_flat.view(env_batch_size, -1) # (num_envs, 3)

            local_obj_root_vel = global_obj_lin.view(env_batch_size, -1, 3)
            local_obj_root_vel_flat = my_quat_rotate(heading_inv_rot.view(-1, 4), local_obj_root_vel.view(-1, 3))
            self._obs_local_obj_vel = local_obj_root_vel_flat.view(env_batch_size, -1) # (num_envs, 3)


            self.dif_object_position = ref_obj_pos - global_obj_pos
            self.dif_object_rotation = quat_mul(ref_obj_rot, quat_conjugate(self.global_obj_rot, w_last=True), w_last=True)
            self.dif_object_lin_vel = ref_obj_lin - global_obj_lin
            self.dif_object_ang_vel = ref_obj_ang - global_obj_ang
            
            dif_global_obj_pos_for_obs_compute = ref_obj_pos.view(env_batch_size, -1, 3) - global_obj_pos.view(env_batch_size, -1, 3)
            dif_local_obj_pos_flat = my_quat_rotate(heading_inv_rot.view(-1, 4), dif_global_obj_pos_for_obs_compute.view(-1, 3))
            self._obs_dif_local_obj_pos = dif_local_obj_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
            
            global_ref_obj_pos = ref_obj_pos.view(env_batch_size, -1, 3) - root_pos.view(env_batch_size, 1, 3)  # preserves the body position
            local_ref_obj_pos_flat = my_quat_rotate(heading_inv_rot.view(-1, 4), global_ref_obj_pos.view(-1, 3))
            self._obs_local_ref_obj_pos = local_ref_obj_pos_flat.view(env_batch_size, -1) # (num_envs, num_rigid_bodies*3)
            
        pass
    
    def get_gravity_orientation(self, quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation

    def step(self, actions_scaled):
        if self.config.robot.hybrid_control_mode:
            motion_times = torch.tensor((self.episode_length_buf + 1) * self.dt + self.motion_start_times) # next frames so +1 
            motion_res = self._motion_lib.get_motion_state([0], motion_times, offset=None)

            self.ref_joint_pos = motion_res["dof_pos"] #
            self.hybrid_action = np.zeros((self.num_envs, self.num_actions),dtype=np.float32)
            self.ppo_action_indices = [i for i in range(self.num_actions) if i not in self.open_dof_indices]
            self.hybrid_action[:,self.open_dof_indices] = self.ref_joint_pos[:,self.open_dof_indices]
            self.hybrid_action[:,self.rl_action_indices] = actions_scaled
        else:
            self.hybrid_action = actions_scaled

        for i in range(self.control_decimation):
            step_start = time.time()
            qj = self.mjdata.qpos[7:7+ self.num_actions]   
            dqj = self.mjdata.qvel[6:6+ self.num_actions]

            torque = self.pd_control(self.p_gains, self.hybrid_action[0], self.target_dof_pos, qj, self.d_gains, dqj, self.torque_limits)

            self.mjdata.ctrl[:] = torque
            mujoco.mj_step(self.mjmodel, self.mjdata)

            time_until_next_step = self.mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            self.viewer.sync()
        self.episode_length_buf += 1
        
        self._pre_compute_observations_callback()
        self.compute_observation()
        
        clip_obs = self.config.env.config.normalization.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

        for key in self.history_handler.history.keys():
            self.history_handler.add(key, self.hist_obs_dict[key])

    def pd_control(self, kps, actions_scaled, default_angles, dof_pos, kds, dof_vel, torque_limits):
        pos = actions_scaled + default_angles
        torques = kps  *(pos - dof_pos) - kds  * dof_vel
        torques = torch.from_numpy(torques)
        torque_limits = torch.from_numpy(torque_limits)
        return torch.clip(torques, -torque_limits, torque_limits)
    
    # ========== humanoid obs =============
    def _get_obs_base_lin_vel(self,):
        return self.base_lin_vel
    
    def _get_obs_base_ang_vel(self,):
        return self.base_ang_vel
    
    def _get_obs_projected_gravity(self,):
        return self.projected_gravity
    
    def _get_obs_dof_pos(self,):
        return self.dof_pos - self.default_dof_pos
    
    def _get_obs_dof_vel(self,):
        return self.dof_vel

    def _get_obs_local_tar_pos(self,):
        return self._obs_local_obj_pos.squeeze()

    def _get_obs_local_tar_vel(self,):
        return self._obs_local_obj_vel.squeeze()
    
    def _get_obs_tar_rot(self,):
        return self.global_obj_rot.squeeze()

    def _get_obs_ref_motion_phase(self):
        if self.config.obs.obs_dims.ref_motion_phase == 1:
            return self._ref_motion_phase
        elif self.config.obs.obs_dims.ref_motion_phase == 4:
            return self.motions_phase.squeeze()
    
    def _get_obs_actions(self):
        return self.action_torch
    
    def _get_obs_history_actor(self,):
        assert "history_actor" in self.config.obs.obs_auxiliary.keys()
        history_config = self.config.obs.obs_auxiliary['history_actor']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).squeeze()
    
    # ========== object obs ==============
    def _get_obs_dif_local_obj_pos(self):
        return self._obs_dif_local_obj_pos.squeeze()
    
    def _get_obs_local_ref_obj_pos(self):
        return self._obs_local_ref_obj_pos.squeeze()
    
    # ========== key callback =============
    def key_call_back(self, keycode):
        
        if chr(keycode) == "T":
            # TODO
            time_step = 0
        elif chr(keycode) == "R":
            print("Reset")
            self.episode_length_buf = 0
            self.record_count = 0
            
            self.reset(self.load_obj_flag)
            
            self.history_handler.reset([0])
            self.action =np.zeros(self.rl_action_nums, dtype=np.float32)
        elif chr(keycode) == " ":
            # TODO
            self.pause = not self.pause
            print("Paused")
        else:
            print("not mapped", chr(keycode))

# ============ Class MujocoDeployer END ==================


@hydra.main(config_path="../../../humanoidverse/data/key_config_model_file", config_name="base_eval")
def main(override_config: OmegaConf):
    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config_mujoco.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config_mujoco.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config_mujoco path: {config_path}")

        if has_config:
            logger.info(f"Loading training config_mujoco file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config
            
    from humanoidverse.utils.helpers import pre_process_config
    pre_process_config(config)

    mujoco_deplayer = MujocoDeployer(config)
    
    mujoco_deplayer.run()
    



if __name__ == "__main__":
    main()