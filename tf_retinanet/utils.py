import numpy as np

def to_classes(labels):
  # TODO: optimize performance and convert to tensorflow
  data = {}
  index = 0
  classes = np.zeros(shape=(labels.shape[0], ))
  for i in range(labels.shape[0]):
    name = labels[i]
    if not name in data:
      data[name] = index
      index += 1
    classes[i] = data[name]
  return classes, i
