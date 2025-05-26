export CUDA_VISIBLE_DEVICES=0
python humanoidverse/eval_agent.py \
+checkpoint=/home/dmu00/LDY/ASAP-main/logs/MotionTracking/20250526_072323-Start_MultiTask_OriginEP_43Dof_TrackID_bigPD-motion_tracking-g1_29dof_with_hand/model_1200.pt \
+num_envs=4 \
#+headless=True
