
import numpy as np
import casadi                                                                       
import meshcat.geometry as mg
print("ok")
from datetime import datetime 

import pinocchio as pin                             
import time
from pinocchio import casadi as cpin    
from pinocchio.visualize import MeshcatVisualizer   
import os
import sys
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import copy
parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)
np.set_printoptions(precision=5, suppress=True, linewidth=200)
_global_ik_solver = None

def init_ik_worker():
    global _global_ik_solver
    _global_ik_solver = G1_29_ArmIK(Unit_Test=False, Visualization=False)
    #print(f"子进程 {os.getpid()} 求解器初始化完成")
robot = pin.RobotWrapper.BuildFromURDF('humanoidverse/data/robots/g1/g1_29dof_with_hand.urdf', 'humanoidverse/data/robots/g1/')

class WeightedMovingFilter:
    def __init__(self, weights, data_size = 14):
        self._window_size = len(weights)
        self._weights = np.array(weights)
        assert np.isclose(np.sum(self._weights), 1.0), "[WeightedMovingFilter] the sum of weights list must be 1.0!"
        self._data_size = data_size
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue = []

    def _apply_filter(self):
        if len(self._data_queue) < self._window_size:
            return self._data_queue[-1]

        data_array = np.array(self._data_queue)
        temp_filtered_data = np.zeros(self._data_size)
        for i in range(self._data_size):
            temp_filtered_data[i] = np.convolve(data_array[:, i], self._weights, mode='valid')[-1]
        
        return temp_filtered_data

    def add_data(self, new_data):
        assert len(new_data) == self._data_size

        if len(self._data_queue) > 0 and np.array_equal(new_data, self._data_queue[-1]):
            return  # skip duplicate data
        
        if len(self._data_queue) >= self._window_size:
            self._data_queue.pop(0)

        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    @property
    def filtered_data(self):
        return self._filtered_data

class G1_29_ArmIK:
    def __init__(self, Unit_Test = False, Visualization = False):
        #

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        if not self.Unit_Test:
            self.robot = robot
        else:
            self.robot = robot

        self.mixed_jointsToLockIDs = [
                                        "left_hip_pitch_joint" ,
                                        "left_hip_roll_joint" ,
                                        "left_hip_yaw_joint" ,
                                        "left_knee_joint" ,
                                        "left_ankle_pitch_joint" ,
                                        "left_ankle_roll_joint" ,
                                        "right_hip_pitch_joint" ,
                                        "right_hip_roll_joint" ,
                                        "right_hip_yaw_joint" ,
                                        "right_knee_joint" ,
                                        "right_ankle_pitch_joint" ,
                                        "right_ankle_roll_joint" ,
                                        "waist_yaw_joint" ,
                                        "waist_roll_joint" ,
                                        "waist_pitch_joint" ,
                                        
                                        "left_hand_thumb_0_joint" ,
                                        "left_hand_thumb_1_joint" ,
                                        "left_hand_thumb_2_joint" ,
                                        "left_hand_middle_0_joint" ,
                                        "left_hand_middle_1_joint" ,
                                        "left_hand_index_0_joint" ,
                                        "left_hand_index_1_joint" ,
                                        
                                        "right_hand_thumb_0_joint" ,
                                        "right_hand_thumb_1_joint" ,
                                        "right_hand_thumb_2_joint" ,
                                        "right_hand_index_0_joint" ,
                                        "right_hand_index_1_joint" ,
                                        "right_hand_middle_0_joint",
                                        "right_hand_middle_1_joint"
                                    ]

        init_dof  = np.array([0.0] * 43)
        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration= init_dof,
        )
        self.reduced_robot.model.addFrame(
            pin.Frame('L_ee',
                      self.reduced_robot.model.getJointId('left_wrist_yaw_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.05,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )

        self.reduced_robot.model.addFrame(
            pin.Frame('R_ee',
                      self.reduced_robot.model.getJointId('right_wrist_yaw_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.05,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3,3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3,3]
                )
            ],
        )

        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3,:3].T)
                )
            ],
        )
        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)
        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)
        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-6
            },
            'print_time':False,# print or not
            'calc_lam_p':False # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq) #
        self.smooth_filter_t = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        if self.Visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
            self.vis.initViewer(open=False) 
            self.vis.loadViewerModel("pinocchio") 
            self.vis.displayFrames(True, frame_ids=[107, 108], axis_length = 0.15, axis_width = 5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ['L_ee_target', 'R_ee_target']
            FRAME_AXIS_POSITIONS = (
                np.array([[0, 0, 0], [1, 0, 0],
                          [0, 0, 0], [0, 1, 0],
                          [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
            )
            FRAME_AXIS_COLORS = (
                np.array([[1, 0, 0], [1, 0.6, 0],
                          [0, 1, 0], [0.6, 1, 0],
                          [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            )
            axis_length = 0.3
            axis_width = 20
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

    def solve_ik(self, left_wrist, right_wrist, origin_q, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        if self.Visualization:
            self.vis.viewer['L_ee_target'].set_transform(left_wrist)   # for visualization
            self.vis.viewer['R_ee_target'].set_transform(right_wrist)  # for visualization

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            self.smooth_filter_t.add_data(sol_q)
            sol_q = self.smooth_filter_t.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, 0
        
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter_t.add_data(sol_q)
            sol_q = self.smooth_filter_t.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            print(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}")
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return origin_q, 1
        
    def get_interDR_dof14(self, env_ids, new_left_pos, new_right_pos, origin_q):
        # initial positon
        ik_nums = len(env_ids)
        new_left_pos[:,-1] = new_left_pos[:,-1] + 0.054 
        new_right_pos[:,-1] = new_right_pos[:,-1] + 0.054

        sol_q_all = np.zeros((ik_nums,14))
        ik_succeed_id = []

        for env in range(ik_nums): 
            L_tf_target = pin.SE3(pin.Quaternion(1,0,0,0.0),new_left_pos[env],) #0.05
            R_tf_target = pin.SE3(pin.Quaternion(1,0,0,0.),new_right_pos[env],) #-0.2
            step = 0
            while True:
                sol_q, failed_flag = self.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, current_lr_arm_motor_q=origin_q[env] ,origin_q=origin_q[env])
                step += 1
                if step > 10 or failed_flag:
                    break
            if not failed_flag: ik_succeed_id.append(env)
            sol_q_all[env] = sol_q
        return sol_q_all, np.array(ik_succeed_id)
    
def get_interDR_dof14_parallel(env_ids, new_left_pos, new_right_pos, current_q_list):
    ik_nums = len(env_ids)
    sol_q_all = np.zeros((ik_nums, 14))
    updated_filter_states = [None] * ik_nums
    tasks = []
    for i, env_id in enumerate(range(ik_nums)):
        tasks.append((
            env_id,
            new_left_pos[i].copy(),
            new_right_pos[i].copy(),
            current_q_list[i].copy(),
        ))
    with mp.Pool(
        processes=min(os.cpu_count(), ik_nums)#, initializer=init_worker
    ) as pool:
        results = pool.map(_solve_single_env, tasks)
        
        for env_id, sol_q in results:
            sol_q_all[env_id] = sol_q
            #updated_filter_states[env_id] = new_filter_state
    return sol_q_all, updated_filter_states
    
def _solve_single_env(args):
    """任务处理函数（模块级函数）"""
    global _global_ik_solver
    env_id, new_left_pos_env, new_right_pos_env, current_q_env = args
    
    new_left_pos_env[-1] += 0.054
    new_right_pos_env[-1] += 0.054
    
    L_tf_target = pin.SE3(pin.Quaternion(1, 0, 0, 0), new_left_pos_env)
    R_tf_target = pin.SE3(pin.Quaternion(1, 0, 0, 0), new_right_pos_env)
    for i in range(5):
        sol_q, failed_flag = _global_ik_solver.solve_ik(
            L_tf_target.homogeneous,
            R_tf_target.homogeneous,
            #current_lr_arm_motor_q=current_q_env
        )
        if failed_flag: break
    return env_id, sol_q

def test_ik():
    t = time.time()

    #arm_ik = G1_29_ArmIK(Unit_Test = False, Visualization = False)
    sample = np.array([
              [[0.3039, 0.1004, 0.1969],[0.3550, -0.2057,  0.2821]],

              ])
    sample = sample.repeat(200, axis=0)
    batchsize = sample.shape[0]
    current_q_list = [np.zeros(14).copy() for _ in range(batchsize)]
    filter_list = []
    # for _ in range(batchsize):
    #     weights = np.array([0.4, 0.3, 0.2, 0.1])  # 滤波器的权重
    #     data_size = 14  # 滤波器的数据大小
    #     filter_instance = WeightedMovingFilter(weights, data_size)
    #     filter_list.append(filter_instance)

    sol_q_batch1, _ = get_interDR_dof14_parallel(
        env_ids=np.array(range(batchsize)),
        new_left_pos=sample[:,0,:],  # shape: [N, 3]
        new_right_pos=sample[:,1,:], # shape: [N, 3]
        current_q_list=current_q_list,
    )
    # sol_q_batch1, _ = get_interDR_dof14_parallel(
    #     env_ids=np.array(range(batchsize)),
    #     new_left_pos=sample[:,0,:],  # shape: [N, 3]
    #     new_right_pos=sample[:,1,:], # shape: [N, 3]
    #     current_q_list=current_q_list,
    # )

    print(batchsize, "Time:", time.time() - t)
    print(sol_q_batch1)
if __name__ == "__main__":

    initializer=init_ik_worker()
    t = time.time()

    #arm_ik = G1_29_ArmIK(Unit_Test = False, Visualization = False)
    sample = np.array([
              [[0.3039, 0.1004, 0.1969],[0.3550, -0.2057,  0.2821]],

              ])
    sample = sample.repeat(200, axis=0)
    batchsize = sample.shape[0]
    current_q_list = [np.zeros(14).copy() for _ in range(batchsize)]
    filter_list = []
    # for _ in range(batchsize):
    #     weights = np.array([0.4, 0.3, 0.2, 0.1])  # 滤波器的权重
    #     data_size = 14  # 滤波器的数据大小
    #     filter_instance = WeightedMovingFilter(weights, data_size)
    #     filter_list.append(filter_instance)

    sol_q_batch1, _ = get_interDR_dof14_parallel(
        env_ids=np.array(range(batchsize)),
        new_left_pos=sample[:,0,:],  # shape: [N, 3]
        new_right_pos=sample[:,1,:], # shape: [N, 3]
        current_q_list=current_q_list,
    )
    # sol_q_batch1, _ = get_interDR_dof14_parallel(
    #     env_ids=np.array(range(batchsize)),
    #     new_left_pos=sample[:,0,:],  # shape: [N, 3]
    #     new_right_pos=sample[:,1,:], # shape: [N, 3]
    #     current_q_list=current_q_list,
    # )

    print(batchsize, "Time:", time.time() - t)
    print(sol_q_batch1)
