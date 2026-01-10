import onnx

model = onnx.load("models/retinanet_r50_fpn.onnx")
for out in model.graph.output:
    print("   ", out.name, out.type.tensor_type.shape.dim)