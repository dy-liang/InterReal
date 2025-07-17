export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/miniconda3/envs/hvgym1/lib/
export CUDA_VISIBLE_DEVICES=1
python  humanoidverse/eval_agent.py \
+checkpoint=/gemini/code/ASAP/logs/MotionTracking/20250716_203312-Start_43Dof_MultiTask_OpenHand_PDv4_interDR_50x5_Pos_Mov3_nohist-motion_tracking-g1_29dof_with_hand/model_9000.pt \
+num_envs=1 \
+headless=True
#+headless=True -m debugpy --listen 5677 --wait-for-client
