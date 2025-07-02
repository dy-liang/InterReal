import torch
import numpy as np
import pickle
import  joblib
#file="/home/dmu00/LDY/ASAP-main/humanoidverse/data/motions/g1_29dof_anneal_23dof/TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl"
file="/home/dmu00/LDY/Tele2025/HOI_ASAP/humanoidverse/data/motions/g1_43dof/250520_HOIs_43dof.pkl"
data = joblib.load(file)
#print(data)
print(data.keys()) 
data1 = data["largebox"] 
print(data1.keys()) 
print(data1["obj_state"][32,0:3])
print("left dof pos: ", data1["dof"][1,[19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42]],"left dof pos: ", data1["dof"][1,29:36])    #(119, 3)    0.8272517

file = "/home/dmu00/LDY/Tele2025/HOI_ASAP/humanoidverse/data/motions/g1_43dof/250520_HOIs_43dof+aug.pkl"
data = joblib.load(file)
#print(data)
print(data.keys()) 
data1 = data["largebox"] 
print(data1.keys()) 
print(data1["obj_state"][31,0:3])    #(119, 3)    0.8272517
print(np.where(abs(data1["dof_pos"]-1.3115)<0.0001))
print("left dof pos: ", data1["dof"][30,15:22],"left dof pos: ", data1["dof"][30,29:36])    #(119, 3)    0.8272517

# print(data["terminate"])             #(119, 4)    
# print(data["obj_contact"])             #(119, 4)     

# #print(data["root_trans_offset"])    #(119, 3)    
# #print(data["root_rot"])    #(119, 3)    

# print(data.keys())                        #dict_keys(['root_trans_offset', 'pose_aa', 'dof', 'root_rot', 'smpl_joints', 'fps'])
# print(data["root_trans_offset"].shape)    #(119, 3)     
# print(data["pose_aa"].shape)                #(119, 27, 3) / (128, 1, 3) 每个关节3维，对应Unitree G1的27个驱动关节（含身体+手指）
# print(data["dof"].shape)                  #(119, 23)    
# print(data["root_rot"].shape)             #(119, 4)   

# #print(data["smpl_joints"].shape)          #(119, 24, 3) 
# print(data["fps"])                  #30
# print(data["terminate"])             #(119, 4)    


# dict_keys(['root_trans_offset', 'pose_aa', 'dof', 'root_rot', 'smpl_joints', 'fps'])
# (119, 3)
# (119, 27, 3)
# (119, 23)
# (119, 4)
# (119, 24, 3)
# 30
