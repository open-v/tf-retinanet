import tensorflow as tf
from tf_retinanet.builders import (
  backbone_builder,
  model_builder,
  loss_builder,
)

def build(backbone, num_classes, num_anchors, shape=(None, None, 3), inputs=None):
  if not inputs:
    inputs = tf.keras.Input(shape=shape)
  backbone = backbone_builder.build(backbone, inputs=inputs)

  models = [
    ('localization', model_builder.build_localization_model(4, num_anchors)),
    ('classification', model_builder.build_classification_model(num_classes, num_anchors)),
  ]

  model = model_builder.build(backbone, models)
  return model
