export CUDA_VISIBLE_DEVICES=0
python humanoidverse/eval_agent.py \
+checkpoint=/home/dmu00/LDY/ASAP-main/logs/MotionTracking/20250522_173407-Start_MultiTask_OriginEP_43Dof_TrackID-motion_tracking-g1_29dof_with_hand/model_6800.pt \
+num_envs=4 \
#+headless=True
