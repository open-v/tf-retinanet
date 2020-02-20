import tensorflow as tf
import tf_retinanet
from tensorflow.keras import layers

class Anchors(layers.Layer):
  def __init__(self, sizes=None, strides=None, ratios=None, scales=None, **kwargs):
    super(Anchors, self).__init__(**kwargs)
    self.sizes = sizes if sizes else tf_retinanet.configs.config.anchors.sizes
    self.strides = strides if strides else tf_retinanet.configs.config.anchors.strides
    self.ratios = ratios if ratios else tf_retinanet.configs.config.anchors.ratios
    self.scales = scales if scales else tf_retinanet.configs.config.anchors.scales
    self.anchors = tf.stack([
      self._build(self.sizes[i], self.ratios, self.scales)
      for i in range(self.sizes.shape[0])
    ])

  def call(self, features):
    return tf.concat([
      f.tile(tf.expand_dims(self._shift(tf.shape(feature)[1:3], self.strides[i], self.anchors[i]), axis=0),
          (tf.shape(feature)[0], 1, 1))
      for i, feature in enumerate(features)
    ], axis=1)

  def _build(self, base_size, ratios, scales):
    shapes = tf.constant([ratios.shape[0], scales.shape[0]])
    scales = tf.reshape(base_size * tf.tile(tf.expand_dims(scales, axis=0), (1, shapes[0])), (-1, ))
    ratios = tf.repeat(ratios, shapes[1])
    width  = tf.sqrt(scales * scales / ratios)
    height = width * ratios
    return tf.transpose(tf.stack([0 - width, 0 - height, width, height], axis=0)) * 0.5

  def _shift(self, shape, stride, anchors):
    shape   = tf.cast(shape, dtype=tf.float32)
    grid    = tf.meshgrid(
      tf.range(shape[1], dtype=tf.float32) + 0.5,
      tf.range(shape[0], dtype=tf.float32) + 0.5
    )
    shift_x = tf.reshape(grid[0] * stride, (-1, ))
    shift_y = tf.reshape(grid[1] * stride, (-1, ))
    shifts  = tf.transpose(tf.stack([shift_x, shift_y, shift_x, shift_y], axis=0))
    k = tf.shape(shifts)[0]
    q = tf.shape(anchors)[0]
    shifted_anchors = tf.reshape(anchors, (1, q, 4)) + tf.reshape(shifts, (k, 1, 4))
    return tf.reshape(shifted_anchors, (k * q, 4))
