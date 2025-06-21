import xml.etree.ElementTree as ET
import numpy as np

def calculate_joint_distances(urdf_string, joint_chain):
    """
    计算URDF中指定关节链的相邻关节间距
    
    参数:
    urdf_string: URDF字符串
    joint_chain: 关节名称列表，按顺序排列
    
    返回:
    字典: {关节对: 距离}
    """
    # 解析URDF
    root = ET.fromstring(urdf_string)
    
    # 存储关节信息
    joint_info = {}
    
    # 收集所有关节信息
    for joint in root.findall('joint'):
        joint_name = joint.get('name')
        origin = joint.find('origin')
        if origin is not None:
            xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
            joint_info[joint_name] = {
                'xyz': xyz,
                'parent': joint.find('parent').get('link'),
                'child': joint.find('child').get('link')
            }
    
    # 计算相邻关节间距
    distances = {}
    
    for i in range(1, len(joint_chain)):
        prev_joint = joint_chain[i-1]
        curr_joint = joint_chain[i]
        
        # 获取当前关节的位置（相对于父连杆）
        if curr_joint not in joint_info:
            raise ValueError(f"关节 '{curr_joint}' 未在URDF中找到")
        
        # 当前关节的位置向量
        curr_pos = np.array(joint_info[curr_joint]['xyz'])
        
        # 计算距离（欧几里得距离）
        distance = np.linalg.norm(curr_pos)
        joint_pair = f"{prev_joint} → {curr_joint}"
        distances[joint_pair] = distance
    
    return distances

# 给定的关节链
joint_chain = [
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint'
]

# 从文件读取URDF内容（实际使用时替换为您的URDF文件路径）
with open('humanoidverse/data/robots/g1/g1_29dof_with_hand.urdf', 'r') as f:
    urdf_string = f.read()

# 计算距离
distances = calculate_joint_distances(urdf_string, joint_chain)

# 打印结果
print("相邻关节间距计算结果:")
for joint_pair, distance in distances.items():
    print(f"{joint_pair}: {distance:.6f} m")

# left_shoulder_pitch_joint → left_shoulder_roll_joint: 0.040439 m
# left_shoulder_roll_joint → left_shoulder_yaw_joint: 0.103388 m
# left_shoulder_yaw_joint → left_elbow_joint: 0.082050 m
# left_elbow_joint → left_wrist_roll_joint: 0.100516 m
# left_wrist_roll_joint → left_wrist_pitch_joint: 0.038000 m
# left_wrist_pitch_joint → left_wrist_yaw_joint: 0.046000 m