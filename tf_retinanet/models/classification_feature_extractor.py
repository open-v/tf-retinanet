import tf_retinanet
from tensorflow.keras import layers
from .feature_extractor import FeatureExtractor

class ClassificationFeatureExtractor(FeatureExtractor):
  def __init__(self, num_classes=None, num_anchors=None, name='classification', **kwargs):
    super(ClassificationFeatureExtractor, self).__init__(name=name, **kwargs)
    if num_classes is None:
      num_classes = tf_retinanet.configs.config.num_classes
    if num_anchors is None:
      num_anchors = tf_retinanet.configs.config.anchors.num_anchors
    self.conv_5 = layers.Conv2D(num_classes * num_anchors, name='{}_conv_5'.format(name), **self.options)
    self.reshape = layers.Reshape((-1, num_classes), name='{}_reshape'.format(name))
    self.sigmoid = layers.Activation('sigmoid', name='{}_sigmoid'.format(name))

  def call(self, x):
    x = super(ClassificationFeatureExtractor, self).call(x)
    x = self.conv_5(x)
    x = self.reshape(x)
    return self.sigmoid(x)
