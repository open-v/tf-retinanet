import tensorflow as tf

def resize(source, target):
  shape = tf.keras.backend.shape(target)
  return tf.image.resize(source, (shape[1], shape[2]))

