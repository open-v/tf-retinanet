import cv2
import numpy as np

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
