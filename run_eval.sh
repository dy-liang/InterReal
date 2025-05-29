export CUDA_VISIBLE_DEVICES=1
python  humanoidverse/eval_agent.py \
+checkpoint=/home/dmu00/LDY/Tele2025/HOI_ASAP/humanoidverse/data/key_config_model_file/43dof_config_model_Mig_4000phase/model/model_17900.pt \
+num_envs=1 \
#+headless=True -m debugpy --listen 5677 --wait-for-client
