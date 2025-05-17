export CUDA_VISIBLE_DEVICES=1
python humanoidverse/eval_agent.py \
+checkpoint=/home/dmu00/LDY/ASAP-main/logs/MotionTracking/20250517_060043-MotionTracking_CR7-motion_tracking-g1_29dof_anneal_23dof/model_10000.pt \
+num_envs=4 \
#+headless=True
