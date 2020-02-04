import numpy as np
import tensorflow as tf

def build_anchors(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    ratios=np.array([0.5, 1, 2], tf.keras.backend.floatx()),
    scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], tf.keras.backend.floatx())):
  config = {
    'sizes': sizes,
    'strides': strides,
    'ratios': ratios,
    'scales': scales
  }
  config['num_anchors'] = len(config['ratios']) * len(config['scales'])
  return config

def build(num_classes, learning_rate=1e-5, **kwargs):
  return {
    'learning_rate': learning_rate,
    'num_classes': num_classes,
    'anchors': build_anchors(**kwargs)
  }