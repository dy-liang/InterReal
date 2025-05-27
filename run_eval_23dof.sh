##### checkout to the branch "hois_23/43dof_v2" first #####

export CUDA_VISIBLE_DEVICES=0
python  humanoidverse/eval_agent.py \
+checkpoint=/home/dyliang/LDY/HOI_ASAP/pre_train/model/model_17200.pt \
+num_envs=4 \
