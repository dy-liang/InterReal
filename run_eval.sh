export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/miniconda3/envs/hvgym1/lib/
export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1
python  humanoidverse/eval_agent.py \
+checkpoint=logs/MotionTracking/20250730_235736-Start_43Dof_MultiTask_OpenHand_PDv4_interDR_50x5_Pos_Mov3_AutoR_origin-motion_tracking-g1_29dof_with_hand/model_9500.pt \
+num_envs=1 \
+headless=True
#+headless=True -m debugpy --listen 5677 --wait-for-client
