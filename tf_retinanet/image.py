import cv2
import numpy as np
from PIL import Image
from .transform import transform_aabb, apply_transform

def read(path):
  image = np.asarray(Image.open(path).convert('RGB'))
  return image[:, :, ::-1].copy()

def resize(image, min_side=800, max_side=1333):
  (rows, cols, _) = image.shape
  scale = min_side / min(rows, cols)
  mx = max(rows, cols)
  if mx * scale > max_side:
    scale = max_side / mx
  image = cv2.resize(image, None, fx=scale, fy=scale)
  return image, scale

def preprocess(image, mode='coffe'):
  assert(mode == 'coffe' or mode=='tf')
  image = image.astype(np.float32)
  modes = {
    'tf': lambda x: (x / 127.5) - 1.0,
    'coffe': lambda x: x - [103.939, 116.779, 123.68]
  }
  return modes[mode](image)

def transform(image, bboxes, mode='coffe', min_side=800, max_side=1333, transform=None, **kwargs):
  if transform:
    image = apply_transform(transform, image, **kwargs)
    bboxes = bboxes.copy()
    for index in range(bboxes.shape[0]):
      bboxes[index, :] = transform_aabb(transform, bboxes[index, :])
  image = preprocess(image, mode=mode)
  image, scale = resize(image, min_side=min_side, max_side=max_side)
  bboxes = bboxes * scale
  return image, bboxes
