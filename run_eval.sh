export CUDA_VISIBLE_DEVICES=1
python humanoidverse/eval_agent.py \
+checkpoint=/home/dmu00/LDY/ASAP-main/logs/MotionTracking/20250520_153337-23URDF_OriginEP_23Dof-motion_tracking-g1_29dof_anneal_23dof/model_0.pt \
+num_envs=1 \
#+headless=True
