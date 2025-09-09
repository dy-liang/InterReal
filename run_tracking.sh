#export ORION_UMA=1
export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/miniconda3/envs/hvgym1/lib/
export CUDA_VISIBLE_DEVICES=0
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=motion_tracking \
+domain_rand=domain_rand_hois \
+rewards=motion_tracking/reward_motion_tracking_dm_2real \
+robot=g1/g1_29dof_with_hand \
+terrain=terrain_locomotion_plane \
+obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
num_envs=4096 \
seed=0 \
project_name=MotionTracking \
experiment_name=Start_43Dof_MultiTask_OpenHand_PDv4_interDR_50x5_Pos_Mov3_AutoR_A_W_nextSR_newSAC_iter50_TTF_scale_ZSI_decay_bigM_allBody_TTT_s0 \
robot.motion.motion_file="humanoidverse/data/motions/g1_43dof/motion_largebox_0d12_5x5_dr_v3.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.adaptive_tracking_sigma.enable=True \
rewards.meta_adaptive=True \
rewards.reward_penalty_degree=0.00001 \
env.config.resample_motion_when_training=False \
env.config.termination.terminate_when_motion_far=True \
env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.45 \
env.config.termination_curriculum.terminate_when_motion_far_threshold_max=0.45 \
env.config.termination_curriculum.terminate_when_motion_far_initial_threshold=0.45 \
env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
robot.asset.self_collisions=0 \
robot.hybrid_control_mode=True \
headless=True
 
#"humanoidverse/data/motions/g1_43dof/250520_HOIs_43dof.pkl"  motion_largebox_0dot1_5x5_dr
# -m debugpy --listen 5678 --wait-for-client
