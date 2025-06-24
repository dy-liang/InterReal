export CUDA_VISIBLE_DEVICES=0
python  humanoidverse/eval_agent.py \
+checkpoint=/home/dyliang/LDY/HOI_ASAP/logs/MotionTracking/20250621_181343-Start_43Dof_MultiTask_OpenHand_PDv4_NewRewObs_Moig+noTrackElbowShouder+TaskID+TaskPhase_ObjR_interDR_50x5_1.5-motion_tracking-g1_29dof_with_hand/model_1200.pt \
+num_envs=1 \
+headless=False
#+headless=True -m debugpy --listen 5677 --wait-for-client
