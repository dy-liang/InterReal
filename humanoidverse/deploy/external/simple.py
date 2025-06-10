
# Grammar: ckpt=_external_xxxx

from humanoidverse.deploy import *
from typing import Dict
import numpy as np
from pathlib import Path
import onnxruntime as ort


class SimplePolicyObs:
    def __init__(self, path: Path, motion_len: float):
        assert path.suffix == '.onnx', f"File {path} is not a .onnx file."
        
        self.session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])  # 使用CPU

        self.actor_dim = 380
        self.action_dim = 23

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        example_input = np.random.randn(1, self.actor_dim).astype(np.float32)
        try_inferr = self.session.run([self.output_name], {self.input_name: example_input})
        assert try_inferr[0].shape == (1, self.action_dim), f"Action shape {try_inferr[0].shape} does not match expected shape (1, {self.action_dim})."
        
        
        self.timer = 0
        self.obs_buf = np.zeros(( self.actor_dim),dtype=np.float32)
        self._hist_len = 4
        self.history = np.zeros((self._hist_len, 23+3+23+23+3+1),dtype=np.float32)
        self.motion_len = motion_len # for guitar
            # 23+3+23+23 = 3*23+3 = 72
            # 3+1 = 4
        
        # Obs Arrangement:
            # obs_keys: ['actions', 'base_ang_vel', 'dof_pos', 'dof_vel', 'history_actor', 'projected_gravity', 'ref_motion_phase']
            # obs shape: {'actions': torch.Size([23]), 'base_ang_vel': torch.Size([3]), 'dof_pos': torch.Size([23]), 'dof_vel': torch.Size([23]), 'history_actor': torch.Size([304]), 'projected_gravity': torch.Size([3]), 'ref_motion_phase': torch.Size([1])}
            #   obs_scales:
                    # base_lin_vel: 2.0
                    # base_ang_vel: 0.25
                    # projected_gravity: 1.0
                    # dof_pos: 1.0
                    # dof_vel: 0.05
                    # actions: 1.0
                    # dif_local_rigid_body_pos: 1.0
                    # local_ref_rigid_body_pos: 1.0
                    # ref_motion_phase: 1.0
                    # history_actor: 1.0
                    # history_critic: 1.0
                    # dr_base_com: 1.0
                    # dr_link_mass: 1.0
                    # dr_kp: 1.0
                    # dr_kd: 1.0
                    # dr_friction: 1.0
                    # dr_ctrl_delay: 1.0
            
            # history_actor: (step)
            #   base_ang_vel: 4
            #   projected_gravity: 4
            #   dof_pos: 4
            #   dof_vel: 4
            #   actions: 4
            #   ref_motion_phase: 4

            # history_tensors: {'actions': torch.Size([1, 92]), 'base_ang_vel': torch.Size([1, 12]), 'dof_pos': torch.Size([1, 92]), 'dof_vel': torch.Size([1, 92]), 'projected_gravity': torch.Size([1, 12]), 'ref_motion_phase': torch.Size([1, 4])}
            
            # history update:
                # def add(self, key: str, value: Tensor):
                #     assert key in self.history.keys(), f"Key {key} not found in history"
                #     val = self.history[key].clone()
                #     self.history[key][:, 1:] = val[:, :-1]
                #     self.history[key][:, 0] = value.clone()
        
    def obs_fn(self,robot: URCIRobot)->np.ndarray:
        # print('Simple ',robot.timer, self.timer)
        
        ### 1. Ticking timer. If timer is not equal, that means the policy is newly SWITCHED in.
        
        if robot.timer != self.timer or robot.motion_len != self.motion_len:
            # self.policy.reset_memory()
            self.timer = robot.timer
            self.history *= 0
            robot.motion_len = self.motion_len
            
            
        ### 2. Update observation buffer.
        motion_time = (self.timer) * robot.dt 
        ref_motion_phase = motion_time / self.motion_len
        robot.ref_motion_phase = ref_motion_phase
        step_obs = np.concatenate([
            robot.act * 1.0,
            robot.omega * 0.25,
            (robot.q - robot.dof_init_pose)*1.0,
            robot.dq * 0.05,
            
            robot.gvec * 1.0,
            np.array([ref_motion_phase])
        ])
        
        hist_obs_split = np.split(self.history, [23, 23+3, 23+3+23, 23+3+23+23, 23+3+23+23+3], axis=1)
        hist_obs = np.concatenate([x.reshape(-1) for x in hist_obs_split], axis=0)
        
        self.obs_buf[:72] = step_obs[:72]
        self.obs_buf[72:72+304] = hist_obs
        self.obs_buf[72+304:] = step_obs[72:]
        
        ### 3. Update History.
        self.history = np.roll(self.history, shift=1, axis=0)
        self.history[0] = step_obs
        
        # breakpoint()
            
        self.timer += 1
        
        return self.obs_buf
    
    
    
    def act_fn(self,obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        # return np.zeros((1, self.action_dim))
        # assert obs.shape == (1, actor_dim), f"Observation shape {obs.shape} does not match expected shape (1, {actor_dim})."
        result = self.session.run([self.output_name], obs_dict)
        # obs = obs_dict[input_name]
        # result = session.run([output_name], {input_name: obs_dict})
        # result = session.run([output_name], {input_name: obs})
        return result[0]


def GetSimplePolicy(name: str, config: CfgType)->URCIPolicyObs:
    # Usage: $EVALMJC=_external_zero
    
    if name == 'guitar':
        ckpt_path = Path("/home/bai/ASAP/logs_real_0415_0416/0416/20250413_025119-ABLReDR1_Guitar2_tr2pf-motion_tracking-g1_23dof_lock_wrist/exported/model_65500.onnx")
        motion_len = 9.7
    elif name == 'dance150d700':
        ckpt_path = Path("/home/bai/ASAP/logs_real_0415_0416/0416/20250413_025119-ABLReDR1_Guitar2_tr2pf-motion_tracking-g1_23dof_lock_wrist/exported/model_65500.onnx")
        motion_len = 9.7
        
    else:
        raise ValueError(f"Unknown policy name: {name}")
        
    policy = SimplePolicyObs(ckpt_path, motion_len)
        
    return (policy.obs_fn, policy.act_fn)