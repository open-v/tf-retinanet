from tensorflow.keras import layers
from .feature_extractor import FeatureExtractor

class LocalizationFeatureExtractor(FeatureExtractor):
  def __init__(self, num_values, num_anchors=None, name='localization', **kwargs):
    super(LocalizationFeatureExtractor, self).__init__(name=name, **kwargs)
    self.conv_5 = layers.Conv2D(num_values * num_anchors, name='{}_conv_5'.format(name), **self.options)
    self.reshape = layers.Reshape((-1, num_values), name='{}_reshape'.format(name))

  def call(self, x):
    x = super(LocalizationFeatureExtractor, self).call(x)
    x = self.conv_5(x)
    return self.reshape(x)
