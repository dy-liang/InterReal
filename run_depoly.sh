python humanoidverse/urci.py \
+simulator=real \
+checkpoint=humanoidverse/data/key_config_model_file/23dof_model/model_0.onnx \
+simulator.config.name=real \
+simulator.config.sim.fps=500 \
+simulator.config.sim.control_decimation=10 \
+robot.asset.xml_file="humanoidverse/data/robots/g1/g1_29dof_with_hand.xml"