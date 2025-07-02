# python humanoidverse/urci.py \
# +simulator=real \
# +checkpoint=humanoidverse/data/key_config_model_file/43dof_wObjPos/model_25100.onnx \
# +simulator.config.name=real \
# +simulator.config.sim.fps=500 \
# +simulator.config.sim.control_decimation=10 \
# +robot.asset.xml_file="humanoidverse/data/robots/g1/g1_29dof_with_hand.xml" \
# +deploy.net=eno1 \
# +deploy.Dex3=True \
# +deploy.DEBUG=True
export DISPLAY=:1.0
#export CKPTS=[humanoidverse/data/key_config_model_file/43dof_aug/model_49200.onnx,humanoidverse/data/key_config_model_file/43dof_aug/model_49200.onnx]

export CKPTS=[humanoidverse/data/key_config_model_file/43dof_woObjPos/model_25100.onnx,humanoidverse/data/key_config_model_file/43dof_woObjPos/model_25100.onnx]
python humanoidverse/urci.py \
+simulator=mujoco \
+checkpoint=$CKPTS \
+simulator.config.name=mujoco \
+simulator.config.sim.fps=200 \
+simulator.config.sim.control_decimation=4 \
+robot.asset.xml_file="g1/g1_29dof_with_hand.xml" \

#humanoidverse/data/key_config_model_file/23dof_model/model_0.onnx  -m debugpy --listen 5678 --wait-for-client 