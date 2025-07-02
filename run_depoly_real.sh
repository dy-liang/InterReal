export CKPTS=[humanoidverse/data/key_config_model_file/43dof_wObjPos/model_25100.onnx]

python humanoidverse/urci.py \
+simulator=real \
+checkpoint=$CKPTS \
+simulator.config.name=real \
+simulator.config.sim.fps=500 \
+simulator.config.sim.control_decimation=10 \
+robot.asset.xml_file="humanoidverse/data/robots/g1/g1_29dof_with_hand.xml" \
+deploy.net=eno1 \
+deploy.Dex3=True \
+deploy.DEBUG=True

#humanoidverse/data/key_config_model_file/23dof_model/model_0.onnx 