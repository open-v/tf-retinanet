import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Add, Activation

class PyramidFeatureExtractor(tf.keras.Model):
  def __init__(self, feature_size=256, **kwargs):
    super(PyramidFeatureExtractor, self).__init__(**kwargs)
    # pyramid feature 3
    self.pyram_3_reduced = Conv2D(feature_size, kernel_size=1, strides=1, padding='same',
        name='pyram_3_reduced')
    self.pyram_3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='pyram_3')
    # pyramid feature 2
    self.pyram_2_reduced = Conv2D(feature_size, kernel_size=1, strides=1, padding='same',
        name='pyram_2_reduced')
    self.pyram_2_merge = Add(name='pyram_2_merge')
    self.pyram_2 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='pyram_2')
    # pyramid feature 1
    self.pyram_1_reduced = Conv2D(feature_size, kernel_size=1, strides=1, padding='same',
        name='pyram_1_reduced')
    self.pyram_1_merge = Add(name='pyram_1_merge')
    self.pyram_1 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='pyram_1')
    # pyramid feature 4
    self.pyram_4 = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='pyram_4')
    # pyramid feature 5
    self.pyram_5_relu = Activation('relu', name='pyram_5_relu')
    self.pyram_5 = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='pyram_5')
  
  def call(self, features, **kwargs):
    f1, f2, f3 = features
    x3 = self.pyram_3_reduced(f3)
    f2_shape = tf.shape(f2)
    x3_resized = tf.image.resize(x3, (f2_shape[1], f2_shape[2]))
    x3 = self.pyram_3(x3)

    x2 = self.pyram_2_reduced(f2)
    x2 = self.pyram_2_merge([x3_resized, x2])
    f1_shape = tf.shape(f1)
    x2_resized = tf.image.resize(x2, (f1_shape[1], f1_shape[2]))
    x2 = self.pyram_2(x2)

    x1 = self.pyram_1_reduced(f1)
    x1 = self.pyram_1_merge([x2_resized, x1])
    x1 = self.pyram_1(x1)

    x4 = self.pyram_4(f3)
    x5 = self.pyram_5_relu(x4)
    x5 = self.pyram_5(x5)
    return [x1, x2, x3, x4, x5]
