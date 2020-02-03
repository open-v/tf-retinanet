import tensorflow as tf
from tf_retinanet.builders import pyramid_builder

def build_localization_model(num_values, num_anchors, name='localization', **kwargs):
  inputs, outputs = pyramid_builder.build_baseline_pyramid_block(
    num_values, num_anchors=num_anchors, name=name, **kwargs)
  return tf.keras.Model(inputs=inputs, outputs=outputs, name=name + '_model')

def build_classification_model(num_classes, num_anchors, name='classification', **kwargs):
  inputs, outputs = pyramid_builder.build_baseline_pyramid_block(
    num_classes, num_anchors=num_anchors, name=name, **kwargs)
  outputs = tf.keras.layers.Activation('sigmoid', name=name + '_sigmoid')(outputs)
  return tf.keras.Model(inputs=inputs, outputs=outputs, name=name + '_model')

def build(backbone, models):
  def _build(name, model, features):
    return tf.keras.layers.Concatenate(axis=1, name=name)([
      model(feature) for feature in features
    ])
  features = pyramid_builder.build_pyramid_features(backbone)
  pyramids = [_build(name, model, features) for name, model in models]
  return tf.keras.Model(inputs=backbone.inputs, outputs=pyramids, name=backbone.name)