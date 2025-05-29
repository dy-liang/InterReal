#export DISPLAY="192.168.155.28:0.0" 
export CUDA_VISIBLE_DEVICES=1
python  humanoidverse/eval_agent.py \
+checkpoint=humanoidverse/data/key_config_model_file/43dof_config_model+Mig_0000/model/model_20000.pt \
+num_envs=4 \
#+headless=True -m debugpy --listen 5677 --wait-for-client
