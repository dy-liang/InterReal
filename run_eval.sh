export CUDA_VISIBLE_DEVICES=1
python humanoidverse/eval_agent.py \
+checkpoint=/home/dmu00/LDY/ASAP-main/logs/MotionTracking/20250523_172723-Start_MultiTask_OriginEP_23Dof_TrackID_bigPD-motion_tracking-g1_29dof_anneal_23dof/model_1500.pt \
+num_envs=4 \
#+headless=True
