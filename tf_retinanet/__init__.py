import tensorflow as tf
from . import builders
from . import configs
from . import layers
from . import losses
from . import models
from . import data
from . import image
from . import visualization
from tqdm import tqdm
from .builders.model_builder import build
from . import anchors

@tf.function
def train(model, dataset, num_classes, epochs=5, lr=1e-5, checkpoints=None):
  optimizer = tf.keras.optimizers.Adam(lr=lr)
  for i in range(epochs):
    dataset = dataset.shuffle(1000)
    # training
    for images, boxes, labels in tqdm(dataset, desc='Epoch {} of {}'.format(i + 1, epochs)):
      with tf.GradientTape() as tape:
        localization, classification = model(images, training=True)
        smooth = losses.smooth_l1(boxes, localization)
        focal  = losses.focal(labels, classification)
      gradients = tape.gradient([smooth, focal], model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def detection(model, images, **kwargs):
  images = tf.cast(images, dtype=tf.float32)
  return model(images, detection=True, **kwargs)
