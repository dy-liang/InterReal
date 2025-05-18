export CUDA_VISIBLE_DEVICES=1
python humanoidverse/eval_agent.py \
+checkpoint=/home/dmu00/LDY/ASAP-main/logs/MotionTracking/20250518_192923-Start_NewURDF_OriginEP_43Dof-motion_tracking-g1_29dof_with_hand/model_0.pt \
+num_envs=1 \
#+headless=True
