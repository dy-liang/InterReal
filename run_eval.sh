export CUDA_VISIBLE_DEVICES=1
python humanoidverse/eval_agent.py \
+checkpoint=/home/dmu00/LDY/ASAP-main/logs/MotionTracking/20250523_144809-Start_MultiTask_OriginEP_43Dof_TrackID_bigPD-motion_tracking-g1_29dof_with_hand/model_600.pt \
+num_envs=1 \
#+headless=True
