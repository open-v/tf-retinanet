import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers

class FeatureExtractor(tf.keras.Model):
  def __init__(self, feature_size=256, kernel_size=3, strides=1, padding='same',
      kernel_initializer=None, bias_initializer=None, name='feature_extractor', **kwargs):
    super(FeatureExtractor, self).__init__(**kwargs)
    if kernel_initializer is None:
      kernel_initializer = initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
    if bias_initializer is None:
      bias_initializer = initializers.Zeros()
    self.options = {'kernel_size': kernel_size,'strides': strides, 'padding': padding,
      'kernel_initializer': kernel_initializer, 'bias_initializer': bias_initializer}
    self.conv_1 = layers.Conv2D(feature_size, activation='relu', name='{}_conv_1'.format(name), **self.options)
    self.conv_2 = layers.Conv2D(feature_size, activation='relu', name='{}_conv_2'.format(name), **self.options)
    self.conv_3 = layers.Conv2D(feature_size, activation='relu', name='{}_conv_3'.format(name), **self.options)
    self.conv_4 = layers.Conv2D(feature_size, activation='relu', name='{}_conv_4'.format(name), **self.options)

  def call(self, x, **kwargs):
    x = self.conv_1(x)
    x = self.conv_2(x)
    x = self.conv_3(x)
    return self.conv_4(x)
