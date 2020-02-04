import tensorflow as tf
import contextlib2
from glob import glob

class Shards:
  def __init__(self, path, num_shards=10):
    self.stack = contextlib2.ExitStack()
    self.filenames = ['{}-{:05d}-of-{:05d}'.format(path, idx, num_shards) for idx in range(num_shards)]
    self.shards = [
      self.stack.enter_context(tf.io.TFRecordWriter(filename))
      for filename in self.filenames
    ]
    self.index = 0
    self.num_shards = num_shards

  def write(self, sample):
    index = self.index % self.num_shards
    self.shards[index].write(sample.SerializeToString())
    self.index += 1

  def __enter__(self):
    return self

  def __exit__(self, _type, value, traceback):
    self.stack.__exit__(_type, value, traceback)

def read(path):
  def _parse(example):
    description = {
      'images': tf.io.FixedLenFeature((), tf.string),
      'target': tf.io.FixedLenFeature((), tf.string),
    }
    sample = tf.io.parse_single_example(example, description)
    image = tf.io.parse_tensor(sample['images'], out_type=tf.float64)
    target = tf.io.parse_tensor(sample['target'], out_type=tf.float64)
    return image, target

  files = glob(path)
  data = tf.data.TFRecordDataset(files)
  data = data.map(_parse)
  return data
