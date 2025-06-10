

from humanoidverse.deploy import *
from typing import Dict
import onnxruntime as ort
import numpy as np

# python humanoidverse/urci.py +simulator=mujoco +deploy.net=eno1 +checkpoint=_external_urghomie

class ONNXPolicy:
    def __init__(self, onnx_path):
        # 创建推理会话
        self.session = ort.InferenceSession(onnx_path)
        
        # 打印输入输出信息用于调试
        print("===== Model Input Details =====")
        for i, inp in enumerate(self.session.get_inputs()):
            print(f"Input {i}: Name={inp.name}, Shape={inp.shape}, Type={inp.type}")
        
        print("\n===== Model Output Details =====")
        for i, out in enumerate(self.session.get_outputs()):
            print(f"Output {i}: Name={out.name}, Shape={out.shape}, Type={out.type}")
        
        # 初始化记忆状态
        self.reset_memory()
    
    def reset_memory(self):
        """根据模型实际输入初始化记忆状态"""
        input_details = self.session.get_inputs()
        
        # 检查是否有状态输入
        if len(input_details) < 3:
            raise RuntimeError("Model does not have expected state inputs (requires at least 3 inputs)")
        
        # 获取隐藏状态和细胞状态的形状
        # ONNX Runtime返回的shape是元组，例如 (1, 2, 64)
        self.h_shape = input_details[1].shape  # hidden_state形状
        self.c_shape = input_details[2].shape  # cell_state形状
        
        # 初始化状态
        # print(input_details[1])
        # print(f"Hidden state shape: {self.h_shape}, Cell state shape: {self.c_shape}")
        self.h_state = np.zeros(self.h_shape, dtype=np.float32)
        self.c_state = np.zeros(self.c_shape, dtype=np.float32)
    
    def predict(self, observation):
        # 准备输入
        inputs = {
            'input': np.array(observation, dtype=np.float32).reshape(1, -1),
            'hidden_state': self.h_state,
            'cell_state': self.c_state
        }
        
        # 执行推理
        output, new_h, new_c = self.session.run(
            ['output', 'new_hidden', 'new_cell'],
            inputs
        )
        
        # 更新状态
        self.h_state, self.c_state = new_h, new_c
        # print(self.h_state, self.c_state)
        
        return output

def GetURGPolicyLXZ_12DoF()->URCIPolicyObs:
    # Usage: $EVALMJC=_external_urglxz
    
    # ckptpath = "/home/bcj/URG_BEAM/urg_beam/logs/g1_15dof/exported/Apr10_20-55-37_g1_first_true_default_angle-true_arm_cul-15dof/policy_lstm_5500.pt"
    # ckptpath = "/home/bcj/HOMI-RL-lxz/HOMIE-RL/legged_gym/logs/exported/policies/policy.pt"
    ckptpath = "/home/bai/ASAP/logs/External/0420_policy_lstm_6000.pt"
    # ckptpath = "/home/bcj/ASAP-wjx/logs/External/policy_lstm_27500.pt"
    ckptpath_onnx = '/home/bcj/ASAP-wjx/logs/External/0420_best_policy_lstm_6000.onnx'
    USE_ONNX = False
    
    
    import torch, numpy as np
    
    default_angles = np.array([-0.1,  0.0,  0.0,  0.3, -0.2, 0.0, 
                              -0.1,  0.0,  0.0,  0.3, -0.2, 0.0]
                              +[ 0, 0, 0,
                                     0.2,  0.2, 0, 0.9,
                                     0.2, -0.2, 0, 0.9], dtype=np.float32)
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    action_scale = 0.25
    cmd_scale = np.array([2.0, 2.0, 0.25], dtype=np.float32)
    max_cmd = np.array([0.8, 0.5, 1.57], dtype=np.float32)

    num_urg_actions = 23
    num_obs = 9+num_urg_actions*2+ 14  # 17
    # num_obs = 47
    
    class URGPolicyObs:
        def __init__(self):
            self.obs_buf = np.zeros((num_obs), dtype=np.float32)
            
            # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = torch.device("cpu")
            print("URG: device=",self.device)
            if USE_ONNX:
                self.deployer = ONNXPolicy(ckptpath_onnx)
            else:
                self.policy = torch.jit.load(ckptpath, map_location=self.device)
                self.policy.eval()
                
            self.timer = 0
            
            self.phase_counter = 0
            
            
        def obs(self, robot: URCIRobot)->np.ndarray:
            # print(robot.timer, self.timer)
            
            # phase = (robot.timer * robot.dt ) % 0.8 / 0.8
            is_zero_cmd = float(np.linalg.norm(robot.cmd[:2])<0.1)
            # # phase*= (1- is_zero_cmd)
            
            period = 0.8
            if np.linalg.norm(robot.cmd[:2]) < 0.1:
                self.phase_counter = 0
            else:
                self.phase_counter += 1
            
            count = self.phase_counter * robot.dt   #### 写的也有问题
            
            if np.linalg.norm(robot.cmd[:2]) < 0.1:
                phase = 0
                offset = 0
            else:
                phase = count % period / period
                offset = 0.5
            phase_left = phase
            phase_right = (phase + offset) % 1
            left_sin_phase = np.sin(2 * np.pi * phase_left)
            right_sin_phase = np.sin(2 * np.pi * phase_right)

            self.obs_buf[0:3] = robot.omega * ang_vel_scale
            self.obs_buf[3:6] = robot.gvec
            self.obs_buf[6:9] = robot.cmd[:3] * cmd_scale
            self.obs_buf[6:8] *= (1- is_zero_cmd)
            
            self.obs_buf[9:9+num_urg_actions] = (robot.q[:num_urg_actions] - default_angles[:num_urg_actions]) * dof_pos_scale
            self.obs_buf[9+num_urg_actions:9+num_urg_actions*2] = robot.dq[:num_urg_actions] * dof_vel_scale
            
            self.obs_buf[9+num_urg_actions*2:9+num_urg_actions*2+12] = robot.act[:12]
            # self.obs_buf[9+num_urg_actions*2:9+num_urg_actions*3] = robot.act[:num_urg_actions]
            
            self.obs_buf[9+num_urg_actions*2+12] = left_sin_phase # np.sin(2 * np.pi * phase)
            self.obs_buf[9+num_urg_actions*2+13] = right_sin_phase # np.cos(2 * np.pi * phase)
            
            if robot.timer != self.timer:
                # self.policy.reset_memory()
                self.timer = robot.timer
            else:
                self.timer += 1
            return self.obs_buf
        
        def act(self, obs_dict: Dict[str, np.ndarray])->np.ndarray:
            
            real_act = np.zeros((1, 23), dtype=np.float32)
            # real_act[0,:12] = self.policy(obs_tensor).detach().cpu().numpy()
            
            if USE_ONNX:
                real_act[0,:12] = self.deployer.predict(self.obs_buf)[0]
            else:
                obs_tensor = torch.from_numpy(self.obs_buf).unsqueeze(0).to(self.device)
                real_act[0,:12] = self.policy(obs_tensor).detach().cpu().numpy()
            # real_act[0,:12] = self.deployer.predict(self.obs_buf)[0]
            
            # real_act[0,:num_urg_actions] = self.policy(obs_tensor).detach().numpy()
            # real_act[0,num_urg_actions:] = default_upper_angles
            
            return real_act
    urg = URGPolicyObs()
    
    return (urg.obs, urg.act)
    ...

def GetURGPolicyLXZ()->URCIPolicyObs:
    # Usage: $EVALMJC=_external_urglxz
    ckptpath = "/home/bcj/URG_BEAM/urg_beam/logs/g1_15dof/exported/Apr10_20-55-37_g1_first_true_default_angle-true_arm_cul-15dof/policy_lstm_5500.pt"
    import torch, numpy as np
    
    default_angles = np.array([-0.1,  0.0,  0.0,  0.3, -0.2, 0.0, 
                              -0.1,  0.0,  0.0,  0.3, -0.2, 0.0]
                              +[ 0, 0, 0,
                                     0.2,  0.2, 0, 0.9,
                                     0.2, -0.2, 0, 0.9], dtype=np.float32)
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    action_scale = 0.25
    cmd_scale = np.array([2.0, 2.0, 0.25], dtype=np.float32)
    max_cmd = np.array([0.8, 0.5, 1.57], dtype=np.float32)

    num_urg_actions = 23
    num_obs = 9+num_urg_actions*2+17
    # num_obs = 47
    
    class URGPolicyObs:
        def __init__(self):
            self.obs_buf = np.zeros((num_obs), dtype=np.float32)
            self.policy = torch.jit.load(ckptpath)
            self.policy.eval()
            self.timer = 0
            
        def obs(self, robot: URCIRobot)->np.ndarray:
            # print(robot.timer, self.timer)
            
            phase = (robot.timer * robot.dt ) % 0.8 / 0.8
            is_zero_cmd = float(np.linalg.norm(robot.cmd[:2])<0.2)
            # phase*= (1- is_zero_cmd)
            
            self.obs_buf[0:3] = robot.omega * ang_vel_scale
            self.obs_buf[3:6] = robot.gvec
            self.obs_buf[6:9] = robot.cmd[:3] * cmd_scale
            self.obs_buf[6:8] *= (1- is_zero_cmd)
            
            self.obs_buf[9:9+num_urg_actions] = (robot.q[:num_urg_actions] - default_angles[:num_urg_actions]) * dof_pos_scale
            self.obs_buf[9+num_urg_actions:9+num_urg_actions*2] = robot.dq[:num_urg_actions] * dof_vel_scale
            
            self.obs_buf[9+num_urg_actions*2:9+num_urg_actions*2+15] = robot.act[:15]
            # self.obs_buf[9+num_urg_actions*2:9+num_urg_actions*3] = robot.act[:num_urg_actions]
            
            self.obs_buf[9+num_urg_actions*2+15] = np.sin(2 * np.pi * phase)
            self.obs_buf[9+num_urg_actions*2+16] = np.cos(2 * np.pi * phase)
            
            if robot.timer != self.timer:
                # self.policy.reset_memory()
                self.timer = robot.timer
            else:
                self.timer += 1
            return self.obs_buf
        
        def act(self, obs_dict: Dict[str, np.ndarray])->np.ndarray:
            obs_tensor = torch.from_numpy(self.obs_buf).unsqueeze(0)
            
            real_act = np.zeros((1, 23), dtype=np.float32)
            real_act[0,:15] = self.policy(obs_tensor).detach().numpy()
            # real_act[0,:num_urg_actions] = self.policy(obs_tensor).detach().numpy()
            # real_act[0,num_urg_actions:] = default_upper_angles
            
            return real_act
    urg = URGPolicyObs()
    
    return (urg.obs, urg.act)
    ...

def GetURGPolicy(name: str)->URCIPolicyObs:
    # Usage: $EVALMJC=_external_urg0
    if name == 'lxz_12dof':
        return GetURGPolicyLXZ_12DoF()
    if name == 'lxz':
        return GetURGPolicyLXZ()
    if name == 'homie':
        return GetURGPolicyLXZ_12DoF()
    assert name == '0'
    # ckptpath="logs/External/_external_urg0/exported/Mar25_22-41-36_kpkdasap_handsdown_H64rnn_RAMC/policy_lstm_1.pt"
    ckptpath = "/home/bcj/URG_BEAM/urg_beam/logs/g1_15dof/exported/Apr10_20-55-37_g1_first_true_default_angle-true_arm_cul-15dof/policy_lstm_5500.pt"
    # ckptpath = "/home/bai/ASAP/logs/External/_external_urg0/exported/lxz_15dof_0411/policy_lstm_1.pt"
    import torch, numpy as np
    
    default_angles = np.array([-0.1,  0.0,  0.0,  0.3, -0.2, 0.0, 
                              -0.1,  0.0,  0.0,  0.3, -0.2, 0.0], dtype=np.float32)
    default_upper_angles = np.array([ 0, 0, 0,
                                     0.2,  0.2, 0, 0.9,
                                     0.2, -0.2, 0, 0.9], dtype=np.float32)
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    action_scale = 0.25
    cmd_scale = np.array([2.0, 2.0, 0.25], dtype=np.float32)
    max_cmd = np.array([0.8, 0.5, 1.57], dtype=np.float32)

    num_urg_actions = 12
    num_obs = 47
    
    class URGPolicyObs:
        def __init__(self):
            self.obs_buf = np.zeros((num_obs), dtype=np.float32)
            self.policy = torch.jit.load(ckptpath)
            self.policy.eval()
            self.timer = 0
            
        def obs(self, robot: URCIRobot)->np.ndarray:
            # print('URG ',robot.timer, self.timer)
            
            phase = (robot.timer * robot.dt ) % 0.8 / 0.8
            is_zero_cmd = float(np.linalg.norm(robot.cmd[:2])<0.2)
            # phase*= (1- is_zero_cmd)
            
            self.obs_buf[0:3] = robot.omega * ang_vel_scale
            self.obs_buf[3:6] = robot.gvec
            self.obs_buf[6:9] = robot.cmd[:3] * cmd_scale
            self.obs_buf[6:8] *= (1- is_zero_cmd)
            
            self.obs_buf[9:9+num_urg_actions] = (robot.q[:num_urg_actions] - default_angles) * dof_pos_scale
            self.obs_buf[9+num_urg_actions:9+num_urg_actions*2] = robot.dq[:num_urg_actions] * dof_vel_scale
            
            self.obs_buf[9+num_urg_actions*2:9+num_urg_actions*3] = robot.act[:num_urg_actions]
            
            self.obs_buf[9+num_urg_actions*3] = np.sin(2 * np.pi * phase)
            self.obs_buf[9+num_urg_actions*3+1] = np.cos(2 * np.pi * phase)
            
            if robot.timer != self.timer:
                # self.policy.reset_memory()
                self.timer = robot.timer
            else:
                self.timer += 1
            return self.obs_buf
        
        def act(self, obs_dict: Dict[str, np.ndarray])->np.ndarray:
            obs_tensor = torch.from_numpy(self.obs_buf).unsqueeze(0)
            
            real_act = np.zeros((1, 23), dtype=np.float32)
            real_act[0,:num_urg_actions] = self.policy(obs_tensor).detach().numpy()
            # real_act[0,num_urg_actions:] = default_upper_angles
            
            return real_act
    urg = URGPolicyObs()
    
    return (urg.obs, urg.act)