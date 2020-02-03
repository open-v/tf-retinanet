import tensorflow as tf

BACKBONES = {
  'mobilenet_v1': (tf.keras.applications.mobilenet.MobileNet,
    ['conv_pw_5_relu', 'conv_pw_11_relu', 'conv_pw_13_relu']),
}

def build(backbone, inputs):
  if backbone in BACKBONES:
    backbone, features = BACKBONES[backbone]
    application = backbone(input_tensor=inputs, include_top=False, pooling=None, weights=None)
    outputs = [application.get_layer(feature).output for feature in features]
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=application.name)
  else:
    raise ValueError('Backbone ({}) is not supported'.format(backbone))
