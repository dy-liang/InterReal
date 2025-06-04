export CUDA_VISIBLE_DEVICES=1
python  humanoidverse/eval_agent.py \
+checkpoint=humanoidverse/data/key_config_model_file/43dof_config_model_Mig_0000+obkR+Ct/model/model_10500.pt \
+num_envs=1 \
#+headless=True -m debugpy --listen 5677 --wait-for-client
