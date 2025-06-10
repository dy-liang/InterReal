##### checkout to the branch "hois_23/43dof_v2" first #####

export CUDA_VISIBLE_DEVICES=1
python  humanoidverse/eval_agent.py \
+checkpoint=/home/dmu00/LDY/Tele2025/HOI_ASAP/logs/MotionTracking/20250610_112358-Start_MultiTask_OriginEP_23Dof_TrackID_bigPD_TrackElbowShoulder-motion_tracking-g1_29dof_anneal_23dof/model_0.pt \
+num_envs=1 \
