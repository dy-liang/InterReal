from typing import Dict, List, Tuple, Callable, Optional, Any, Union, TypeVar
import numpy as np
from omegaconf import OmegaConf, DictConfig, ListConfig
from loguru import logger
from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from humanoidverse.utils.helpers import parse_observation, np2torch, torch2np
import time
import torch
import os
from pathlib import Path
from humanoidverse.data.robots.dtype import RobotExitException
from humanoidverse.utils.motion_lib.hoi_motion_lib_base import MotionLibHoi
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
import joblib
URCIRobotType = TypeVar('URCIRobotType', bound='URCIRobot')
ObsCfg = Union[DictConfig, Callable[[URCIRobotType],np.ndarray]]
URCIPolicyObs = Tuple[ObsCfg, Callable]

CfgType = Union[OmegaConf, ListConfig, DictConfig]


class URCIRobot:
    REAL: bool
    BYPASS_ACT: bool
    SWITCH_EMA: bool
    
    dt: float # big dt, not small dt
    clip_observations: float
    cfg: OmegaConf
    
    q: np.ndarray
    dq: np.ndarray
    # pos: np.ndarray
    # vel: np.ndarray
    quat: np.ndarray  # XYZW
    omega: np.ndarray
    gvec: np.ndarray
    rpy: np.ndarray
    
    act: np.ndarray
    
    _obs_cfg_obs: ObsCfg
    # _obs_cfg_obs: Optional[ObsCfg]=None
    _ref_pid = 0 # reference policy id
                    # Normal pid: (0,1,2,3)
                    # Special pid: (-2, -3, -4)
    _pid_size = -1
    
    def __init__(self, cfg: CfgType):
        self.BYPASS_ACT = cfg.deploy.BYPASS_ACT
        self.SWITCH_EMA = cfg.deploy.SWITCH_EMA

        self.cfg: OmegaConf = cfg
        self._obs_cfg_obs = cfg.obs
        self.device: str = "cpu"
        self.dt: float = cfg.deploy.ctrl_dt
        self.timer: int = 0
        self.load_motion = True
        
        self.num_actions = cfg.robot.actions_dim
        self.heading_cmd = cfg.deploy.heading_cmd   
        self.clip_action_limit: float = cfg.robot.control.action_clip_value
        self.clip_observations: float = cfg.env.config.normalization.clip_observations
        self.action_scale: float = cfg.robot.control.action_scale
        self.dof_idx_23_to_29: List[int] = cfg.deploy.dof_idx_23_to_29
        self.dof_idx_23_of_29: List[int] = cfg.deploy.dof_idx_23_of_29 # trained dof idx in 29
        if self.load_motion:
            data_seq = joblib.load(cfg.env.config.robot.motion.motion_file)
            self.obj_name = list(data_seq.keys())
            self._motion_lib = MotionLibHoi(cfg.env.config.robot.motion, num_envs=1, device=self.device)
            self._motion_lib.load_motions(random_sample=False)
            self.motion_duration = self._motion_lib.get_motion_length()[0]
            self.motion_start_times = 0
            self.use_motion_start_dof_pos = True
            self.open_dof_indices = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42] # 20
            self.hand_indices = [22, 23, 24, 25, 26, 27, 28,  36, 37, 38, 39, 40, 41, 42] #14
            self.robot_29dof_indices = [i for i in range(self.num_actions) if i not in self.hand_indices] #29
            self.rl_action_indices = [i for i in range(self.num_actions) if i not in self.open_dof_indices] #23

        self._make_init_pose()
        self._make_buffer()
        if cfg.log_task_name == "motion_tracking":
            self.is_motion_tracking = True
            self._make_motionlib()
        else:
            self.is_motion_tracking = False
        if 'save_motion' in cfg.env.config:
            self.save_motion = bool(cfg.env.config.save_motion)
            if self.save_motion: self._make_init_save_motion()
        else: 
            self.save_motion = False

    
    def routing(self, cfg_policies: List[URCIPolicyObs]):
        """
            Usage: Input a list of Policy, and the robot can switch between them.
            
            - Policies are indexed by integers (Pid), 0 to len(cfg_policies)-1.
            - special pid: 
                - -2: Reset the robot.
                - 0: Default policy, should be stationary. 
                    - The Robot will switch to this policy once the motion tracking is Done or being Reset.
            - Switching Mechanism:
                - The instance (MuJoCo or Real) should implement the Pid control logic. It can be changed at any time.
                - When the instance want to Reset the robot, it should set the pid to -2.
        """
        self._pid_size = len(cfg_policies)
        self._check_init()
        self.cmd[3]=self.rpy[2]
        cur_pid = -1

        try: 
            while True:
                t1 = time.time()
                
                if cur_pid != self._ref_pid or self._ref_pid == -2:
                    if self._ref_pid == -2:
                        print("")
                        self.Reset()
                        self._ref_pid = 0
                        t1 = time.time()
                        ...
                    
                    
                    self._ref_pid %= self._pid_size
                    assert self._ref_pid >= 0 and self._ref_pid < self._pid_size, f"Invalid policy id: {self._ref_pid}"
                    self.TrySaveMotionFile(pid=cur_pid)       
                    logger.info(f"Switch to the policy {self._ref_pid}")

                    
                    cur_pid = self._ref_pid
                    self.SetObsCfg(cfg_policies[cur_pid][0])
                    policy_fn = cfg_policies[cur_pid][1]
                    if self.SWITCH_EMA:
                        self.old_act = self.act.copy()
                    # print('Debug: ',self.Obs()['actor_obs'])
                    # breakpoint()
                
                action = policy_fn(self.Obs())[0]
                action = self.Check_add_open_actions(action)

                #if self.BYPASS_ACT: action = np.zeros_like(action)
                
                # if self.SWITCH_EMA and self.timer <10:
                #     self.old_act = self.old_act * 0.9 + action * 0.1
                #     action = self.old_act
                    
                self.ApplyAction(action)

                self.TrySaveMotionStep()

                if self.motion_len > 0 and self.ref_motion_phase > 1.0:
                    # self.Reset()
                    if self._ref_pid == 0:
                        self._ref_pid = -2
                    else:
                        self._ref_pid = 0
                    self.TrySaveMotionFile(pid=cur_pid)
                    logger.info("Motion End. Switch to the Default Policy")
                    import sys
                    sys.exit()
                t2 = time.time()
                
                # print(f"t2-t1 = {(t2-t1)*1e3} ms")
                if self.REAL:
                # if True:
                    # print(f"t2-t1 = {(t2-t1)*1e3} ms")
                    remain_dt = self.dt - (t2-t1)
                    if remain_dt > 0:
                        time.sleep(remain_dt)
                    else:
                        logger.warning(f"Warning! delay = {t2-t1} longer than policy_dt = {self.dt} , skip sleeping")
        except RobotExitException as e:
            self.TrySaveMotionFile(pid=cur_pid)
            raise e
        ...
    
    def looping(self, cfg_policy: URCIPolicyObs):
        # TODO:
            # annealing out this function `looping`, merge it into `routing`
        return self.routing([cfg_policy])
        obs_cfg, policy_fn = cfg_policy
        self._check_init()
        self.SetObsCfg(obs_cfg)
        self.cmd[3]=self.rpy[2]

        while True:
            t1 = time.time()
            
            if self._ref_pid == -2:
                self.Reset()
                self._ref_pid = 0
                t1 = time.time()
                ...
            
            action = policy_fn(self.Obs())[0]
            
            if self.BYPASS_ACT: action = np.zeros_like(action)
            
            self.ApplyAction(action)
            
            if self.is_motion_tracking and self.ref_motion_phase > 1.05:
                # breakpoint()
                self.Reset()
            
            t2 = time.time()
            
            if self.REAL:
            # if True:
                # print(f"t2-t1 = {t2-t1}")
                remain_dt = self.dt - (t2-t1)
                if remain_dt > 0:
                    time.sleep(remain_dt)
                else:
                    logger.warning(f"Warning! delay = {t2-t1} longer than policy_dt = {self.dt} , skip sleeping")
            # print(f"time: {t2-t1}")
            # time.sleep(self.dt)
        ...
    
    def _reset(self):
        raise NotImplementedError("Not implemented")
    
    def Reset(self):
        # self.TrySaveMotionFile()
        
        self._reset()
        
        self.act[:] = 0
        self.history_handler.reset([0])
        self.timer: int = 0
        self.cmd: np.ndarray = np.array(self.cfg.deploy.defcmd)
        self.cmd[3]=self.rpy[2]
        
        
        self.UpdateObs()
        
    def _apply_action(self, target_q):
        raise NotImplementedError("Not implemented")
    
    def ApplyAction(self, action): 
        self.timer += 1
        self.act = action.copy()
        target_q = np.clip(action, -self.clip_action_limit, self.clip_action_limit) + self.dof_init_pose

        self._apply_action(target_q)

        self.UpdateObs()

    def Check_add_open_actions(self, action):
        if self.cfg.robot.hybrid_control_mode:
            motion_times = torch.tensor((self.timer + 1) * self.dt + self.motion_start_times) # next frames so +1 
            motion_res = self._motion_lib.get_motion_state([0], motion_times, offset=None)

            # open loop control data
            self.ref_joint_pos = motion_res["dof_pos"][0][self.open_dof_indices] #
            if self.cfg.simulator.config.name == "real":
                hand_motion_dof = self.ref_joint_pos.view(2,-1).cpu().numpy()
                wirst = hand_motion_dof[:,:3]
                thumb = hand_motion_dof[:,7:]
                middle = hand_motion_dof[:,5:7]
                index = hand_motion_dof[:,3:5]
                thumb[:,0] = thumb[:,0] + 1.0  #TODO: calibration again
                fix_motion_dof = np.concatenate((wirst,thumb,middle,index),axis=-1).reshape(-1)
            else:
                fix_motion_dof = self.ref_joint_pos

            hybrid_action = np.zeros((self.num_actions),dtype=np.float32)
            hybrid_action[self.open_dof_indices] = fix_motion_dof
            hybrid_action[self.rl_action_indices] = action * self.action_scale
        else:
            hybrid_action = action * self.action_scale
        return hybrid_action

    def Obs(self)->Dict[str, np.ndarray]:
        return {'actor_obs': torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)}
    
    def GetState(self):
        raise NotImplementedError("Not implemented")
    

    def SetObsCfg(self, obs_cfg: ObsCfg):
        if isinstance(obs_cfg, DictConfig):
            self._obs_cfg_obs = obs_cfg
            #self.motion_len = self._obs_cfg_obs.motion_len
            self.motion_len = self.motion_duration
        elif isinstance(obs_cfg, Callable):
            self._obs_cfg_obs = obs_cfg
            self.motion_len = -1
        else:
            raise NotImplementedError("Not implemented")
        
        # self.act[:] = 0
        # self.history_handler.reset([0])
        self.timer: int = 0
        self.cmd: np.ndarray = np.array(self.cfg.deploy.defcmd)
        self.cmd[3]=self.rpy[2]
        if 'ref_motion_phase' in self.history_handler.history.keys():
            self.ref_motion_phase = 0.
            self.history_handler.history['ref_motion_phase']*=0
        self.UpdateObsWoHistory() # Recompute obs with new _obs_cfg_obs
        ...
    
    def UpdateObsWoHistory(self):
        if isinstance(self._obs_cfg_obs, Callable):
            self.obs_buf_dict['actor_obs'] = self._obs_cfg_obs(self)
            return 
        
        obs_cfg_obs = self._obs_cfg_obs
        
        self.obs_buf_dict_raw = {}
        
        noise_extra_scale = 1.
        for obs_key, obs_config in obs_cfg_obs.obs_dict.items():
            if not obs_key=='actor_obs': continue
            self.obs_buf_dict_raw[obs_key] = dict()

            parse_observation(self, obs_config, self.obs_buf_dict_raw[obs_key], obs_cfg_obs.obs_scales, obs_cfg_obs.noise_scales, noise_extra_scale)
        
        self.obs_buf_dict = dict()
        
        for obs_key, obs_config in obs_cfg_obs.obs_dict.items():
            if not obs_key=='actor_obs': continue
            obs_keys = sorted(obs_config)
            self.obs_buf_dict[obs_key] = torch.cat([self.obs_buf_dict_raw[obs_key][key] for key in obs_keys], dim=-1)
            
        clip_obs = self.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            if not obs_key=='actor_obs': continue
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

    def UpdateObsForHistory(self):
        hist_cfg_obs = self.cfg.obs
        
        self.hist_obs_dict = {}
        
        noise_extra_scale = 1. #0
        # Compute history observations
        history_obs_list = self.history_handler.history.keys()
        # print(f"history_obs_list: {history_obs_list}")
        parse_observation(self, history_obs_list, self.hist_obs_dict, hist_cfg_obs.obs_scales, hist_cfg_obs.noise_scales, noise_extra_scale)
        
        for key in self.history_handler.history.keys():
            self.history_handler.add(key, self.hist_obs_dict[key])
    

    def UpdateObs(self):
        self.GetState()
        self.UpdateObsWoHistory()
        self.UpdateObsForHistory()
        
    # TODO: better _check_init
    def _check_init(self):
        assert self.dt is not None, "dt is not set"
        assert self.dt>0 and self.dt < 0.1, "dt is not in the valid range"
        assert self.cfg is not None or not isinstance(self.cfg, OmegaConf), "cfg is not set"
        
        assert self.num_dofs is not None, "num_dofs is not set"
        assert self.num_dofs == 23 or self.cfg.robot.hybrid_control_mode==True, "In policy level, only 23 dofs are supported for now"
        assert self.kp is not None and type(self.kp) == np.ndarray and self.kp.shape == (self.num_dofs,), "kp is not set"
        assert self.kd is not None and type(self.kd) == np.ndarray and self.kd.shape == (self.num_dofs,), "kd is not set"
        
        assert (self.dof_init_pose is not None and type(self.dof_init_pose) == np.ndarray and 
                    self.dof_init_pose.shape == (self.num_dofs,)), "dof_init_pose is not set"
        
        assert self.tau_limit is not None and type(self.tau_limit) == np.ndarray and self.tau_limit.shape == (self.num_dofs,), "tau_limit is not set"
        
        assert self.BYPASS_ACT is not None, "BYPASS_ACT is not set"
        assert self.BYPASS_ACT in [True, False], "BYPASS_ACT is not a boolean, got {self.BYPASS_ACT}"
        
        assert self._pid_size > 0, "pid_size is not correctly set"
    
    def _make_init_pose(self):
        cfg_init_state = self.cfg.robot.init_state
        self.body_names = self.cfg.robot.body_names
        self.dof_names = self.cfg.robot.dof_names
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)

        assert self.num_dofs == 23 or self.cfg.robot.hybrid_control_mode==True, "Only 23 dofs are supported for now" #dyliang 23
        

        dof_init_pose = cfg_init_state.default_joint_angles
        dof_effort_limit_list = self.cfg.robot.dof_effort_limit_list
        
        self.dof_init_pose = np.array([dof_init_pose[name] for name in self.dof_names])
        if self.use_motion_start_dof_pos:
            motion_res = self._motion_lib.get_motion_state([0], torch.tensor(0))
            motion_start_dof_pos = motion_res['dof_pos'][0]
            assert len(motion_start_dof_pos) == 43, "please check 43 motion dofs" #dyliang 23
            self.dof_motion_init_pose = np.array(motion_start_dof_pos)
        if self.cfg.robot.with_open_obs == False:
            self.num_get_dofs = self.num_dofs - len(self.open_dof_indices) #23

        self.tau_limit = np.array(dof_effort_limit_list)
        
        
        self.kp = np.zeros(self.num_dofs) # 43
        self.kd = np.zeros(self.num_dofs)
        
        
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.robot.control.stiffness.keys():
                if dof_name in name:
                    self.kp[i] = self.cfg.robot.control.stiffness[dof_name]
                    self.kd[i] = self.cfg.robot.control.damping[dof_name]
                    found = True
                    logger.debug(f"PD gain of joint {name} were defined, setting them to {self.kp[i]} and {self.kd[i]}")
            if not found:
                raise ValueError(f"PD gain of joint {name} were not defined. Should be defined in the yaml file.")
        
    def _make_buffer(self):
        self.cmd: np.ndarray = np.array(self.cfg.deploy.defcmd)

        self.q = np.zeros(len(self.dof_idx_23_to_29)) #self.num_dofs len(self.dof_idx_23_to_29)
        self.dq = np.zeros(len(self.dof_idx_23_to_29)) #self.num_dofs len(self.dof_idx_23_to_29)
        self.quat = np.zeros(4)  # XYZW
        self.omega = np.zeros(3)
        self.gvec = np.zeros(3)
        self.rpy = np.zeros(3)
        self._obs_local_obj_pos = np.zeros((3)) 
        self._obs_global_obj_pos = np.zeros((3)) 
        self._obs_local_obj_vel = np.zeros((3))
        self._obs_global_obj_vel = np.zeros((3))
        self.global_obj_rot = np.zeros((4))
        self.act = np.zeros(self.num_dofs)
        
        
        self.history_handler = HistoryHandler(1, self.cfg.obs.obs_auxiliary, self.cfg.obs.obs_dims, self.device)
        ...
        
    def _make_motionlib(self):
        self.motion_len =  self.motion_duration #self.cfg.obs.motion_len
        ...
        
        
    # ---- Save Motion System----
        
    def _make_init_save_motion(self):
        save_dir = self.cfg.checkpoint.parent.parent / "motions"
        # os.makedirs(save_dir, exist_ok = True)

        self._save_motion_id = 0
        
        if hasattr(self.cfg, 'dump_motion_name'):
            raise NotImplementedError
            self.save_motion_dir = save_dir / (str(self.cfg.eval_timestamp) + "_" + self.cfg.dump_motion_name)
        else:
            self.save_motion_dir = save_dir / f"{self.cfg.env.config.save_note}_URCI_{type(self).__name__}_{self.cfg.eval_timestamp}"
        self.save_motion_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Save Motion Dir: ", self.save_motion_dir)
        OmegaConf.save(self.cfg, self.save_motion_dir / "config.yaml")
        

        self._dof_axis = torch.load('humanoidverse/utils/motion_lib/dof_axis.pth').numpy()
        self._dof_axis = self._dof_axis.astype(np.float32)

        self.num_augment_joint = len(self.cfg.robot.motion.extend_config)
        self.motions_for_saving: Dict[str, List[np.ndarray]] = {'root_trans_offset':[], 'pose_aa':[], 'dof':[], 'root_rot':[], 'actor_obs':[], 'action':[], 'terminate':[],
                                    'root_lin_vel':[], 'root_ang_vel':[], 'dof_vel':[]}
        self.motion_times_buf = []

        # breakpoint()

    def _get_motion_to_save(self)->Tuple[float, Dict[str, np.ndarray]]:
        raise NotImplementedError("Not implemented")
    
    def TrySaveMotionStep(self):
        if self.save_motion:
            motion_time, motion_data = self._get_motion_to_save()
            self.motion_times_buf.append(motion_time)
            for key, value in motion_data.items():
                if not key in self.motions_for_saving.keys():
                    self.motions_for_saving[key] = []
                    assert isinstance(value, np.ndarray), f"key: {key} is not a np.ndarray"
                self.motions_for_saving[key].append(value)
    
    def TrySaveMotionFile(self, pid: int):
        if self.save_motion:
            import joblib
            from termcolor import colored
            if self.motions_for_saving['root_trans_offset'] == []:
                return
            
            for k, v in self.motions_for_saving.items():
                self.motions_for_saving[k] = np.stack(v).astype(np.float32) # type: ignore
            
            self.motions_for_saving['motion_times'] = np.array(self.motion_times_buf, dtype=np.float32) # type: ignore
            
            dump_data = {}
            keys_to_save = self.motions_for_saving.keys()

            motion_key = f"motion{self._save_motion_id}"
            dump_data[motion_key] = {
                key: self.motions_for_saving[key] for key in keys_to_save
            }
            dump_data[motion_key]['fps'] = 1 / self.dt

            save_path = f'{self.save_motion_dir}/{self._save_motion_id}_pid{pid}_frame{len(self.motions_for_saving["dof"])}_{time.strftime("%Y%m%d_%H%M%S")}.pkl'
            joblib.dump(dump_data, save_path)
            
            logger.info(colored(f"Saved motion data to {save_path}", 'green'))

            self._save_motion_id += 1            
            self.motions_for_saving = {'root_trans_offset':[], 'pose_aa':[], 'dof':[], 'root_rot':[], 'actor_obs':[], 'action':[], 'terminate':[],
                                        'root_lin_vel':[], 'root_ang_vel':[], 'dof_vel':[]}
            self.motion_times_buf = []
        ...
    
    # --------------------------------------------
    
    
    ######################### Observations #########################
    def _get_obs_command_lin_vel(self):
        return np2torch(self.cmd[:2])
    
    def _get_obs_command_ang_vel(self):
        return np2torch(self.cmd[2:3])
    
    def _get_obs_actions(self,):
        return np2torch(self.act[self.rl_action_indices])
    
    def _get_obs_base_pos_z(self,):
        # raise NotImplementedError("Not Implemented")
        return np2torch(self.pos[2:3])
    
    def _get_obs_feet_contact_force(self,):
        raise NotImplementedError("Not implemented")
        return self.data.contact.force[:, :].view(self.num_envs, -1)
          
    
    def _get_obs_base_lin_vel(self,):
        # raise NotImplementedError("Not Implemented")
        # return np2torch(self.vel)
        # print("Warning! base_lin_vel is not allowed to access")
        return torch.zeros(3)
    
    def _get_obs_base_ang_vel(self,):
        return np2torch(self.omega)
    
    def _get_obs_projected_gravity(self,):
        return np2torch(self.gvec)
    
    def _get_obs_dof_pos(self,):
        if len(self.q) == 43: # mujoco
            return np2torch(self.q[self.rl_action_indices] - self.dof_init_pose[self.rl_action_indices])
        return np2torch(self.q[self.dof_idx_23_of_29] - self.dof_init_pose[self.rl_action_indices])
    
    def _get_obs_dof_vel(self,):
        if len(self.dq) == 43: # mujoco
            return np2torch(self.dq[self.rl_action_indices])
        return np2torch(self.dq[self.dof_idx_23_of_29])
    
    def _get_obs_local_tar_pos(self,):
        return np2torch(self._obs_local_obj_pos)

    def _get_obs_global_tar_pos(self,):
        return np2torch(self._obs_global_obj_pos)
    
    def _get_obs_local_tar_vel(self,):
        return np2torch(self._obs_local_obj_vel)

    def _get_obs_global_tar_vel(self,):
        return np2torch(self._obs_global_obj_vel)
    
    def _get_obs_tar_rot(self,):
        return np2torch(self.global_obj_rot)

    def _get_obs_ref_motion_phase(self):
        if self.cfg.obs.obs_dims.ref_motion_phase == 1:
            return np2torch(self.ref_motion_phase)
        elif self.cfg.obs.obs_dims.ref_motion_phase != 1:
            return np2torch(self.motions_phase)
        
    # def _get_obs_ref_motion_phase(self):
    #     # logger.info(f"Phase: {self.ref_motion_phase} | {self.motion_len}")
    #     return torch.tensor(self.ref_motion_phase).reshape(1,)
    
    def _get_obs_dif_local_rigid_body_pos(self):
        raise NotImplementedError("Not implemented")
        return self._obs_dif_local_rigid_body_pos
    
    def _get_obs_local_ref_rigid_body_pos(self):
        raise NotImplementedError("Not implemented")
        return self._obs_local_ref_rigid_body_pos
    
    def _get_obs_vr_3point_pos(self):
        raise NotImplementedError("Not implemented")
        return self._obs_vr_3point_pos
    
    def _get_obs_history(self,):
        obs_cfg_obs = self._obs_cfg_obs
        assert "history" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_short_history(self,):
        obs_cfg_obs = self._obs_cfg_obs
        assert "short_history" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['short_history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_long_history(self,):
        obs_cfg_obs = self._obs_cfg_obs
        assert "long_history" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['long_history']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_history_actor(self,):
        obs_cfg_obs = self._obs_cfg_obs
        assert "history_actor" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['history_actor']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)  # Shape: [4096, history_length*obs_dim]
            history_tensors.append(history_tensor)
            
        # print("history_tensors:", {key: history_tensors[i].shape for i, key in enumerate(sorted(history_config.keys()))})
        # breakpoint()
        return torch.cat(history_tensors, dim=1).reshape(-1)
    
    def _get_obs_history_critic(self,):
        obs_cfg_obs = self._obs_cfg_obs
        assert "history_critic" in obs_cfg_obs.obs_auxiliary.keys()
        history_config = obs_cfg_obs.obs_auxiliary['history_critic']
        history_key_list = history_config.keys()
        history_tensors = []
        for key in sorted(history_config.keys()):
            history_length = history_config[key]
            history_tensor = self.history_handler.query(key)[:, :history_length]
            history_tensor = history_tensor.reshape(history_tensor.shape[0], -1)
            history_tensors.append(history_tensor)
        return torch.cat(history_tensors, dim=1).reshape(-1)