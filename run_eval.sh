export CUDA_VISIBLE_DEVICES=1
python  humanoidverse/eval_agent.py \
+checkpoint=humanoidverse/data/key_config_model_file/43dof_RandomStart_noNoise/model/model_3000.pt \
+num_envs=1 \
#+headless=True -m debugpy --listen 5677 --wait-for-client
