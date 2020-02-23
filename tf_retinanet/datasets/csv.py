import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from .. import image
from ..data import Shards
from ..builders import sample_builder
from ..utils import to_classes

class CSVDatasetBuilder(Shards):
  def __init__(self, path_to, bboxes, labels, num_shards=10, image_min_size=200, image_max_size=300):
    super(CSVDatasetBuilder, self).__init__(path=path_to, num_shards=num_shards)
    self.image_min_size = image_min_size
    self.image_max_size = image_max_size
    self.base_path = os.path.dirname(bboxes)
    self.bboxes = pd.read_csv(bboxes)
    self.bboxes['classes'], self.num_classes = to_classes(self.bboxes['name'].values)
    self.labels = pd.read_csv(labels)

  def convert(self):
    for index, label in tqdm(self.labels.iterrows(), total=self.labels.shape[0], desc='CSV'):
      path = os.path.join(self.base_path, label['filename'])
      try:
        images = image.read(path)
      except:
        print('Could not read image from: {}'.format(path))
        continue
      annotations = self.bboxes[self.bboxes['image_id'] == label['id']]
      bboxes = annotations[['xmin', 'ymin', 'xmax', 'ymax']].values
      images, bboxes = image.transform(images, bboxes,
          min_side=self.image_min_size, max_side=self.image_max_size)
      classes = annotations['classes'].values

      sample = sample_builder.build(images, bboxes, classes, self.num_classes)
      self.write(sample)
