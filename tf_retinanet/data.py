import tensorflow as tf
from glob import glob

def read(path):
  def _parse(example):
    description = {
      'images': tf.io.FixedLenFeature((), tf.string),
      'bboxes': tf.io.FixedLenFeature((), tf.string),
      'classes': tf.io.FixedLenFeature((), tf.string),
    }
    sample = tf.io.parse_single_example(example, description)
    image = tf.io.parse_tensor(sample['images'], out_type=tf.float32)
    bboxes = tf.io.parse_tensor(sample['bboxes'], out_type=tf.float32)
    classes = tf.io.parse_tensor(sample['classes'], out_type=tf.float32)
    return image, bboxes, classes

  files = glob(path)
  data = tf.data.TFRecordDataset(files)
  data = data.map(_parse)
  return data
