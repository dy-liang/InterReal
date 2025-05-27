#### need checkout to the branch "hois_23/43dof_v2" first #####

#export DISPLAY="192.168.155.28:0.0" 
export CUDA_VISIBLE_DEVICES=0
python  humanoidverse/eval_agent.py \
+checkpoint=/home/dyliang/LDY/HOI_ASAP/pre_train/model_17200.pt \
+num_envs=4 \
#+headless=True -m debugpy --listen 5677 --wait-for-client
