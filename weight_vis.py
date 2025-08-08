import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
reward_name = ['teleop_body_position_extend', 'teleop_vr_3point', 'teleop_body_position_feet', 'teleop_body_rotation_extend', 
               'teleop_body_ang_velocity_extend', 'teleop_body_velocity_extend', 'teleop_joint_position', 'teleop_joint_velocity', 
               'teleop_tar_pos', 'teleop_tar_rot', 'teleop_tar_vel', 'interation_graph', 'penalty_torques', 'penalty_action_rate', 
               'penalty_slippage', 'limits_dof_pos', 'limits_dof_vel', 'limits_torque', "termination"]
reward_w = {'teleop_body_position_extend': 0.019999999552965164, 'teleop_vr_3point': 0.031999999284744264, 'teleop_body_position_feet': 0.041999999061226846,
             'teleop_body_rotation_extend': 0.009999999776482582, 'teleop_body_ang_velocity_extend': 0.009999999776482582, 
             'teleop_body_velocity_extend': 0.009999999776482582, 'teleop_joint_position': 0.014999999664723873, 'teleop_joint_velocity': 0.009999999776482582, 
             'teleop_tar_pos': 0.019999999552965164, 'teleop_tar_rot': 0.009999999776482582, 'teleop_tar_vel': 0.019999999552965164,
               'interation_graph': 0.03999999910593033, 'penalty_torques': -1.9999999552965164e-08, 'penalty_action_rate': -0.009999999776482582, 
               'penalty_slippage': -0.019999999552965164, 'limits_dof_pos': -0.19999999552965164, 'limits_dof_vel': -0.09999999776482582, 
               'limits_torque': -0.09999999776482582, 'termination': -3.999999910593033}
# 1. 加载.npy文件
data = np.load('./weight.npy')[:260]  # 替换为实际文件路径
data = data*0.1 + 1.
for i in range(len(reward_name)):
    data[:,i] =  data[:,i] * reward_w[reward_name[i]]
print(data)
# 2. 验证数据形状
print("数据形状:", data.shape)  # 应输出 (270, 19)

# 3. 创建绘图布局
plt.figure(figsize=(18, 20))
plt.suptitle('19 Reward Weights', fontsize=19, fontweight='bold')

# 4. 绘制19个子图
for i in range(19):
    plt.subplot(5, 4, i+1)  # 5行4列布局
    plt.plot(data[:, i], linewidth=1.5, color='blue', alpha=0.8)
    
    # 设置子图标签和格式
    plt.title(f' {reward_name[i]}', fontsize=16)
    plt.xlabel('Phase', fontsize=16)
    plt.ylabel('Weight', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整主标题位置

# 5. 保存并显示图像
plt.savefig('./19_features_plot.png', dpi=300, bbox_inches='tight')
plt.show()