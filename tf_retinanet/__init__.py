import tensorflow as tf
import numpy as np
from tf_retinanet import image as _image
from tf_retinanet import transform as _transform
from tf_retinanet import data
from tf_retinanet.builders.anchors_builder import anchors_for_shape

def resize(source, target):
  shape = tf.keras.backend.shape(target)
  return tf.image.resize(source, (shape[1], shape[2]))

def transform(image, bboxes, mode='coffe', min_side=800, max_side=1333, transform=None, **kwargs):
  if transform:
    image = _transform.apply_transform(transform, image, **kwargs)
    bboxes = bboxes.copy()
    for index in range(bboxes.shape[0]):
      bboxes[index, :] = _transform.transform_aabb(transform, bboxes[index, :])

  image = _image.preprocess(image, mode=mode)
  image, scale = _image.resize(image, min_side=min_side, max_side=max_side)
  bboxes = bboxes * scale
  return image, bboxes

def to_classes(labels):
  data = {}
  index = 0
  classes = np.zeros(shape=(labels.shape[0], ))
  for i in range(labels.shape[0]):
    name = labels[i]
    if not name in data:
      data[name] = index
      index += 1
    classes[i] = data[name]
  return classes

