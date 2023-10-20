'''1. shape inference'''

# 将没有形状信息的onnx模型转换为有形状信息的onnx模型
import onnx

input_onnx_file_path = "mobilenetv2_pt.onnx"
onnx_graph = onnx.load(input_onnx_file_path)
estimated_graph = onnx.shape_inference.infer_shapes(onnx_graph)

output_onnx_file_path = "mobilenetv2_pt_shape.onnx"
onnx.save(estimated_graph, output_onnx_file_path)