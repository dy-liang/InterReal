export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/miniconda3/envs/hvgym1/lib/
export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1
python  humanoidverse/eval_agent.py \
+checkpoint=/gemini/code/HOI_ASAP/logs/MotionTracking/20250904_124603-Start_43Dof_MultiTask_OpenHand_PDv4_interDR_50x5_Pos_Mov3_AutoR_A_W_nextSR_newSAC_iter50_TTF_scale_ZSI_decay_bigM_allBody_TTT_noig_s0-motion_tracking-g1_29dof_with_hand/model_79000.pt \
+meta_checkpoint=/gemini/code/HOI_ASAP/logs/MotionTracking/20250904_124402-Start_43Dof_MultiTask_OpenHand_PDv4_interDR_50x5_Pos_Mov3_AutoR_A_W_nextSR_newSAC_iter50_TTF_scale_ZSI_decay_sM_allBody_TTF_s0-motion_tracking-g1_29dof_with_hand/meta_model_63000.pt \
+num_envs=24 \
+headless=True
#+headless=True -m debugpy --listen 5677 --wait-for-client