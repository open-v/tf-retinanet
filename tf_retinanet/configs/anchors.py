import tensorflow as tf

class AnchorsConfig:
  def __init__(self, sizes=None, strides=None, ratios=None, scales=None):
    sizes   = [32, 64, 128, 256, 512] if sizes is None else sizes
    strides = [8, 16, 32, 64, 128] if strides is None else strides
    ratios  = [0.5, 1, 2] if ratios is None else ratios
    scales  = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)] if scales is None else scales

    self.sizes   = tf.constant(sizes, dtype=tf.float32)
    self.strides = tf.constant(strides, dtype=tf.float32)
    self.ratios  = tf.constant(ratios, dtype=tf.float32)
    self.scales  = tf.constant(scales, dtype=tf.float32)
    self.num_anchors = self.ratios.shape[0] * self.scales.shape[0]

  def get_config(self):
    return {
      'sizes':   self.sizes,
      'strides': self.strides,
      'ratios':  self.ratios,
      'scales':  self.scales
    }

  def __repr__(self):
    return 'Anchors(sizes={}, strides={}, ratios={}, scales={})'.format(
      list(self.sizes.numpy()),
      list(self.strides.numpy()),
      list(self.ratios.numpy()),
      list(self.scales.numpy())
    )
