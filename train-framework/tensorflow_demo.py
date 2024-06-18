'''1. 下载模型'''

# Create the base model from the pre-trained model MobileNet V2
import tensorflow as tf
model = tf.keras.applications.MobileNetV2(weights='imagenet')
model.trainable = False
print(model)
model.summary()


'''2. 转换模型'''

import tf2onnx
spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name="input"),)
# 如果需要可变形状
# spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = model.name + "_tf.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=15, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]
print(output_names)


'''3. 保存模型'''
# 自己创建tf_model目录
# a. 整个模型

# a1. SavedModel 格式
model.save('tf_model/my_model')
# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('tf_model/my_model')
# Check its architecture
new_model.summary()

# a2. HDF5 格式
model.save('tf_model/my_model.h5')
# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('tf_model/my_model.h5')
# Show the model architecture
new_model.summary()

# a3. keras 格式
model.save('tf_model/my_model.keras')
# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('tf_model/my_model.keras')
# Show the model architecture
new_model.summary()

# b. 权重

# ckpt
# Save the weights
model.save_weights('tf_model/my_checkpoint')
# Create a new model instance
new_model = tf.keras.applications.MobileNet(weights=None)
new_model.trainable = False
# Restore the weights
new_model.load_weights('tf_model/my_checkpoint')
new_model.summary()