import tensorflow as tf
import numpy as np
from .. import anchors

def feature_build(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def build(image, bbox, label, num_classes):
  # TODO: convert to tensorflow
  images          = tf.expand_dims(image, axis=0)
  bboxes          = tf.expand_dims(bbox, axis=0)
  labels          = tf.expand_dims(label, axis=0)
  # prepare
  image_anchors     = anchors.for_shape(images[0].shape)
  bboxes, classes   = anchors.anchors_based_targets(image_anchors, images, bboxes, labels, num_classes)
  # convertation
  images = tf.io.serialize_tensor(np.float32(images))
  bboxes = tf.io.serialize_tensor(np.float32(bboxes))
  labels = tf.io.serialize_tensor(np.float32(classes))
  
  feature = {
    'images': feature_build(images),
    'bboxes': feature_build(bboxes),
    'labels': feature_build(labels),
  }
  sample = tf.train.Example(features=tf.train.Features(feature=feature))
  return sample
