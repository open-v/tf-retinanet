import tensorflow as tf

# TODO: mobilenet_v2
# TODO: resnet_v1
# TODO: resnet_v2
# TODO: nasnet_large
# TODO: nasnet_mobile
# TODO: densenet

backbones = {
  'mobilenet_v1': [
    tf.keras.applications.MobileNet, 'conv_pw_5_relu', 'conv_pw_11_relu', 'conv_pw_13_relu']
}

def build(name, shape=(None, None, 3), input_tensor=None, include_top=False, pooling=False, weights=None):
  if input_tensor is None:
    input_tensor = tf.keras.Input(shape=shape)
  if not name in backbones:
    raise ValueError('unsupported backbone')
  Backbone, *features = backbones[name]
  backbone = Backbone(input_tensor=input_tensor, include_top=False, pooling=False, weights=None)
  features = [backbone.get_layer(feature).output for feature in features]
  return tf.keras.Model(inputs=backbone.inputs, outputs=features, name=backbone.name)
