import tensorflow as tf
import keras_tuner as kt

shape = [None, 1030]
placeholder_tensor = tf.TensorSpec(shape, dtype=tf.float32)

path = 'D:\Project\ThesisProject\AutoML\AutoMLProject\gan_model_20241218010914.h5py'
# model = tf.keras.models.load_model()
model = tf.saved_model.load(path)
noise = tf.random.normal(shape=(32,1030))
res = model(noise, training=True)
print(res)