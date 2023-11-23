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