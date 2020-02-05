import numpy as np
import tensorflow as tf
from tf_retinanet.builders import anchors_builder

def build_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def build(images, bboxes, labels, num_classes):
  assert(images.shape[0] == bboxes.shape[0])
  assert(bboxes.shape[0] == labels.shape[0] and bboxes.shape[1] == labels.shape[1])

  max_shape = tuple(max(images[i].shape[x] for i in range(images.shape[0])) for x in range(3))

  _images = np.zeros((images.shape[0],) + max_shape, dtype=np.float32)
  for index in range(images.shape[0]):
    image = images[index]
    _images[index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

  anchors = anchors_builder.anchors_for_shape(max_shape)
  bboxes, classes = anchors_builder.anchor_targets_bbox(anchors, images, bboxes, labels, num_classes)

  images = tf.io.serialize_tensor(np.float32(_images))
  bboxes = tf.io.serialize_tensor(np.float32(bboxes))
  classes = tf.io.serialize_tensor(np.float32(classes))
  
  feature = {
    'images': build_feature(images),
    'bboxes': build_feature(bboxes),
    'classes': build_feature(classes)
  }
  sample = tf.train.Example(features=tf.train.Features(feature=feature))
  return sample
