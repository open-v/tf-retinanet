import cv2

def transform_aabb(matrix, aabb):
  x1, y1, x2, y2 = aabb
  points = matrix.dot([
    [x1, x2, x1, x2],
    [y1, y2, y2, y1],
    [1,  1,  1,  1 ],
  ])
  min_corner = points.min(axis=1)
  max_corner = points.max(axis=1)
  return [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]

def apply_transform(matrix, image, interpolation=cv2.INTER_LINEAR, border_mode=cv2.INTER_NEAREST, border_value=0):
  return cv2.warpAffine(
    image,
    matrix[:2, :],
    dsize=(image.shape[1], image.shape[0]),
    flags=interpolation,
    borderMode=border_mode,
    borderValue=border_value
  )
