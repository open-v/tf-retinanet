import os
import tensorflow as tf
from glob import glob
import contextlib2

class Shards:
  def __init__(self, path, num_shards=10):
    self.stack = contextlib2.ExitStack()
    base_path = os.path.dirname(path)
    if not os.path.isdir(base_path):
      os.makedirs(base_path)
    self.filenames = ['{}-{:05d}-of-{:05d}'.format(path, idx, num_shards) for idx in range(num_shards)]
    self.shards = [
      self.stack.enter_context(tf.io.TFRecordWriter(filename))
      for filename in self.filenames
    ]
    self.index = 0
    self.num_shards = num_shards

  def write(self, sample):
    self.shards[self.index].write(sample.SerializeToString())
    self.index = (self.index + 1) % self.num_shards

  def __enter__(self):
    return self

  def __exit__(self, _type, value, traceback):
    self.stack.__exit__(_type, value, traceback)

def read(path):
  def _parse(example):
    description = {
      'images': tf.io.FixedLenFeature((), tf.string),
      'bboxes': tf.io.FixedLenFeature((), tf.string),
      'labels': tf.io.FixedLenFeature((), tf.string),
    }
    sample = tf.io.parse_single_example(example, description)
    image = tf.io.parse_tensor(sample['images'], out_type=tf.float32)
    bboxes = tf.io.parse_tensor(sample['bboxes'], out_type=tf.float32)
    labels = tf.io.parse_tensor(sample['labels'], out_type=tf.float32)
    return image, bboxes, labels

  files = glob(path)
  data = tf.data.TFRecordDataset(files)
  data = data.map(_parse)
  return data
