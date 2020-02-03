import tensorflow as tf
import tf_retinanet

def build_pyramid_features(backbone, feature_size=256):
  A1, A2, A3 = backbone.outputs

  B3 = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='B3_reduced')(A3)
  B3_resized = tf_retinanet.resize(B3, A2)
  B3 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='B3')(B3)

  B2 = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='B2_reduced')(A2)
  B2 = tf.keras.layers.Add(name='B2_merged')([B3_resized, B2])
  B2_resized = tf_retinanet.resize(B2, A1)
  B2 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='B2')(B2)

  B1 = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='B1_reduced')(A1)
  B1 = tf.keras.layers.Add(name='B1_merged')([B2_resized, B1])
  B1 = tf.keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='B1')(B1)

  B4 = tf.keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='B4')(B3)

  B5 = tf.keras.layers.Activation('relu', name='B4_relu')(B4)
  B5 = tf.keras.layers.Conv2D(feature_size, kernel_size=4, strides=2, padding='same', name='B5')(B5)

  return [B1, B2, B3, B4, B5]

def build_baseline_pyramid_block(n, num_anchors, pyramid_feature_size=256, feature_size=256, name='pyramid'):
  options = {
    'kernel_size': 3,
    'strides': 1,
    'padding': 'same',
    'kernel_initializer': tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
    'bias_initializer': 'zeros'
  }
  inputs = tf.keras.Input(shape=(None, None, pyramid_feature_size))
  x = tf.keras.layers.Conv2D(filters=feature_size, activation='relu', name=name + '_conv_0', **options)(inputs)
  x = tf.keras.layers.Conv2D(filters=feature_size, activation='relu', name=name + '_conv_1', **options)(x)
  x = tf.keras.layers.Conv2D(filters=feature_size, activation='relu', name=name + '_conv_2', **options)(x)
  x = tf.keras.layers.Conv2D(filters=feature_size, activation='relu', name=name + '_conv_3', **options)(x)
  x = tf.keras.layers.Conv2D(n * num_anchors, name='pyramid_' + name, **options)(x)
  x = tf.keras.layers.Reshape((-1, n), name='pyramid_' + name + '_reshape')(x)
  return inputs, x
