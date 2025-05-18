export CUDA_VISIBLE_DEVICES=1
python humanoidverse/eval_agent.py \
+checkpoint=/home/dmu00/LDY/ASAP-main/logs/MotionTracking/20250517_201328-MotionTracking_CR7-motion_tracking-g1_29dof_with_hand/model_11400.pt \
+num_envs=4 \
#+headless=True
