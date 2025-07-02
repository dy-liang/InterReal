export CUDA_VISIBLE_DEVICES=1
python  humanoidverse/eval_agent.py \
+checkpoint=/home/dmu00/LDY/Tele2025/HOI_ASAP/logs/MotionTracking/20250629_223144-Start_43Dof_MultiTask_OpenHand_PDv4_Moig_interDR_50x5_Pos-motion_tracking-g1_29dof_with_hand/model_7600.pt \
+num_envs=1 \
+headless=False
#+headless=True -m debugpy --listen 5677 --wait-for-client
