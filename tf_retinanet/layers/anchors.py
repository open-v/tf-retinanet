import tensorflow as tf
import tf_retinanet
from tensorflow.keras import layers
from .. import anchors

class Anchors(layers.Layer):
  def __init__(self, sizes=None, strides=None, ratios=None, scales=None, **kwargs):
    super(Anchors, self).__init__(**kwargs)
    self.sizes = sizes if sizes else tf_retinanet.configs.config.anchors.sizes
    self.strides = strides if strides else tf_retinanet.configs.config.anchors.strides
    self.ratios = ratios if ratios else tf_retinanet.configs.config.anchors.ratios
    self.scales = scales if scales else tf_retinanet.configs.config.anchors.scales
    self.anchors = tf.stack([
      anchors.build(self.sizes[i], self.ratios, self.scales)
      for i in range(self.sizes.shape[0])
    ])

  def call(self, features):
    return tf.concat([
      f.tile(tf.expand_dims(anchors.shift(tf.shape(feature)[1:3], self.strides[i], self.anchors[i]), axis=0),
          (tf.shape(feature)[0], 1, 1))
      for i, feature in enumerate(features)
    ], axis=1)

