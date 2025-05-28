#export DISPLAY="192.168.155.28:0.0" 
export CUDA_VISIBLE_DEVICES=0
python  humanoidverse/eval_agent.py \
+checkpoint=/home/dyliang/LDY/HOI_ASAP/logs/MotionTracking/20250527_201822-Start_43Dof_MultiTask_OpenHand_PDv4_NewRewObs_Moig+w/oCt+w/oTrackElbowShouder+TaskID+TaskPhase-motion_tracking-g1_29dof_with_hand/model_11200.pt \
+num_envs=1 \
#+headless=True -m debugpy --listen 5677 --wait-for-client
