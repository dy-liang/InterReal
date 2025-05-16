import pickle
import  joblib
#file="/home/dmu00/LDY/ASAP-main/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl"
#file="/home/dmu00/LDY/ASAP-main/humanoidverse/data/motions/g1_43dof/25513_omomo_g1_29dof_with_hand.pkl"
file = "/home/dmu00/LDY/ASAP-main/humanoidverse/data/motions/g1_43dof/25513_omomo_g1_29dof_with_handobj.pkl"
data = joblib.load(file)
print(data)
data = data["motion0"] 
print(data.keys())                        #dict_keys(['root_trans_offset', 'pose_aa', 'dof', 'root_rot', 'smpl_joints', 'fps'])
print(data["root_trans_offset"].shape)    #(119, 3)     ​根关节平移偏移量​（相对初始位置的累积位移）
print(data["pose_aa"].shape)                #(119, 27, 3) / (128, 1, 3) 每个关节3维，对应Unitree G1的27个驱动关节（含身体+手指）
print(data["dof"].shape)                  #(119, 23)    直接映射到Unitree G1的电机驱动指令
print(data["root_rot"].shape)             #(119, 4)     根关节旋转四元数​（全局方向）
#print(data["smpl_joints"].shape)          #(119, 24, 3) 用于可视化与物理仿真验证
print(data["fps"])                  #30
# dict_keys(['root_trans_offset', 'pose_aa', 'dof', 'root_rot', 'smpl_joints', 'fps'])
# (119, 3)
# (119, 27, 3)
# (119, 23)
# (119, 4)
# (119, 24, 3)
# 30