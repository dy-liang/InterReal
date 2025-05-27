#export DISPLAY="192.168.155.28:0.0" 
export CUDA_VISIBLE_DEVICES=0
python  humanoidverse/eval_agent.py \
+checkpoint=pre_train/model/model_10400.pt \
+num_envs=4 \
#+headless=True -m debugpy --listen 5677 --wait-for-client
