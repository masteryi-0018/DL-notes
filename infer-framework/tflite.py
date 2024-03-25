'''1. 将tensor flow模型转换成tflite'''

# a. 转换 SavedModel（推荐）
import tensorflow as tf
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("../train-framework/tf_model/my_model") # path to the SavedModel directory
tflite_model = converter.convert()
# Save the model.
with open('SavedModel.tflite', 'wb') as f:
    f.write(tflite_model)

# b. 转换 Keras 模型
import tensorflow as tf
# Create a model using high-level tf.keras.* APIs
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
model.compile(optimizer='sgd', loss='mean_squared_error') # compile the model
model.fit(x=[-1, 0, 1], y=[-3, -1, 1], epochs=5) # train the model
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_keras_dir")
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save the model.
with open('Keras.tflite', 'wb') as f:
    f.write(tflite_model)

# c. 转换具体函数
import tensorflow as tf
# Create a model using low-level tf.* APIs
class Squared(tf.Module):
    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def __call__(self, x):
        return tf.square(x)
model = Squared()
# (ro run your model) result = Squared(5.0) # This prints "25.0"
# (to generate a SavedModel) tf.saved_model.save(model, "saved_model_tf_dir")
concrete_func = model.__call__.get_concrete_function()
# Convert the model.
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func],
                                                            model)
tflite_model = converter.convert()
# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)


'''2. 量化'''

# 1. 单conv算子实验
import tensorflow as tf
import keras
from keras import layers
model = keras.Sequential()
model.add(keras.Input(shape=(224, 224, 3)))
model.add(layers.Conv2D(filters=64, kernel_size=3, padding='same', activation=None))

model.summary()
# model.save('tf_model') # 新版本tf不可用
model.save('conv.keras')
model.save('conv.h5')
tf.saved_model.save(model, 'conv')

# 2. 单add算子实验
import tensorflow as tf
input1 = tf.keras.layers.Input(shape=(4,))
input2 = tf.keras.layers.Input(shape=(4,))
# equivalent to `added = tf.keras.layers.add([x1, x2])`
added = tf.keras.layers.Add()([input1, input2])
model = tf.keras.models.Model(inputs=[input1, input2], outputs=added)

model.summary()
model.save('add.keras')
model.save('add.h5')
tf.saved_model.save(model, 'add')


# 1. 动态
# from_saved_model在conv2d时出错
converter = tf.lite.TFLiteConverter.from_saved_model("tf_model") # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 默认量化，weight int8，bias float32
tflite_model = converter.convert()
# Save the model.
with open('SavedModel_dyn.tflite', 'wb') as f:
    f.write(tflite_model)


# 2. int8
converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
import numpy as np
def representative_data_gen():
    for _ in range(100):
        data = np.random.rand(1, 224, 224, 3)
        yield [data.astype(np.float32)]
converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()
with open('SavedModel_int8.tflite', 'wb') as f:
    f.write(tflite_model)


# 3. 处理输入输出
converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
import numpy as np
def representative_data_gen():
    for _ in range(100):
        data = np.random.rand(1, 224, 224, 3)
        yield [data.astype(np.float32)]
converter.representative_dataset = representative_data_gen

# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model = converter.convert()
with open('SavedModel_int8_inout.tflite', 'wb') as f:
    f.write(tflite_model)
