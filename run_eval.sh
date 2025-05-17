export CUDA_VISIBLE_DEVICES=1
python humanoidverse/eval_agent.py \
+checkpoint=/home/dmu00/LDY/ASAP-main/logs/MotionTracking/20250517_205135-MotionTracking_CR7-motion_tracking-g1_29dof_with_hand/model_1500.pt \
+num_envs=4 \
#+headless=True
