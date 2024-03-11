# 1. 使用tflite py接口运行模型
import tensorflow as tf
import numpy as np

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="../SavedModel_int8_inout.tflite")
interpreter.allocate_tensors() # invoke前必须调用

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)


# 2. 打印所有算子信息
ops = interpreter._get_ops_details()
all_op = []
for op in ops:
  print(op["op_name"])
  all_op.append(op["op_name"])

print("所有算子：", set(all_op))

'''
details = {
    'index': op_index,
    'op_name': op_name,
    'inputs': op_inputs,
    'outputs': op_outputs,
}
'''


# 3. 打印所有tensor信息
tensors = interpreter.get_tensor_details()
for tensor in tensors:
  print(tensor["name"])

'''
details = {
    'name': tensor_name,
    'index': tensor_index,
    'shape': tensor_size,
    'shape_signature': tensor_size_signature,
    'dtype': tensor_type,
    'quantization': tensor_quantization,
    'quantization_parameters': {
        'scales': tensor_quantization_params[0],
        'zero_points': tensor_quantization_params[1],
        'quantized_dimension': tensor_quantization_params[2],
    },
    'sparsity_parameters': tensor_sparsity_params
}
'''