export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python  humanoidverse/train_agent.py \
+simulator=isaacgym \
+exp=motion_tracking \
+domain_rand=domain_rand_hois \
+rewards=motion_tracking/reward_motion_tracking_dm_2real \
+robot=g1/g1_29dof_with_hand \
+terrain=terrain_locomotion_plane \
+obs=motion_tracking/deepmimic_a2c_nolinvel_LARGEnoise_history \
num_envs=4096 \
project_name=MotionTracking \
experiment_name=Start_43Dof_MultiTask_OpenHand_PDv4_NewRewObs_Moig+noTrackElbowShouder+TaskID+TaskPhase_sub8ID00_DRv4 \
robot.motion.motion_file="humanoidverse/data/motions/g1_43dof/250609_HOIs_ID00_43dof.pkl" \
rewards.reward_penalty_curriculum=True \
rewards.reward_penalty_degree=0.00001 \
env.config.resample_motion_when_training=False \
env.config.termination.terminate_when_motion_far=True \
env.config.termination_curriculum.terminate_when_motion_far_curriculum=True \
env.config.termination_curriculum.terminate_when_motion_far_threshold_min=0.3 \
env.config.termination_curriculum.terminate_when_motion_far_threshold_max=0.5 \
env.config.termination_curriculum.terminate_when_motion_far_initial_threshold=0.45 \
env.config.termination_curriculum.terminate_when_motion_far_curriculum_degree=0.000025 \
robot.asset.self_collisions=0 \
robot.hybrid_control_mode=True \
headless=True
 
#"humanoidverse/data/motions/g1_43dof/250520_HOIs_43dof.pkl" 
# -m debugpy --listen 5678 --wait-for-client
