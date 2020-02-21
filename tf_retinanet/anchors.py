import tensorflow as tf
import tf_retinanet

def guess_shapes(shape, pyramid_levels):
  shape = tf.constant(shape[:2])
  return [(shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]

def for_shape(image_shape, pyramid_levels=[3, 4, 5, 6, 7]):
  sizes = tf_retinanet.configs.config.anchors.sizes
  strides = tf_retinanet.configs.config.anchors.strides
  ratios = tf_retinanet.configs.config.anchors.ratios
  scales = tf_retinanet.configs.config.anchors.scales
  image_shapes = guess_shapes(image_shape, pyramid_levels)
  anchors = tf.concat([
    shift(image_shapes[index], strides[index], build(
      base_size=sizes[index],
      ratios=ratios,
      scales=scales))
    for index, pyramid in enumerate(pyramid_levels)
  ], axis=0)
  return anchors

def shift(shape, stride, anchors):
  shape   = tf.cast(shape, dtype=tf.float32)
  grid    = tf.meshgrid(
    tf.range(shape[1], dtype=tf.float32) + 0.5,
    tf.range(shape[0], dtype=tf.float32) + 0.5
  )
  shift_x = tf.reshape(grid[0] * stride, (-1, ))
  shift_y = tf.reshape(grid[1] * stride, (-1, ))
  shifts  = tf.transpose(tf.stack([shift_x, shift_y, shift_x, shift_y], axis=0))
  k = tf.shape(shifts)[0]
  q = tf.shape(anchors)[0]
  anchors = tf.reshape(anchors, (1, q, 4)) + tf.reshape(shifts, (k, 1, 4))
  return tf.reshape(anchors, (k * q, 4))

def compute_overlap(boxes, query_boxes):
  def _compute_overlap(arguments):
    query_boxes = tf.cast(arguments[0], dtype=tf.float32)
    box_area = (query_boxes[2] - query_boxes[0] + 1) * (query_boxes[3] - query_boxes[1] + 1)
    w = tf.minimum(boxes[:, 2], query_boxes[2]) - tf.maximum(boxes[:, 0], query_boxes[0]) + 1
    h = tf.minimum(boxes[:, 3], query_boxes[3]) - tf.maximum(boxes[:, 1], query_boxes[1]) + 1
    w_bool = tf.cast(tf.greater(w, 0), dtype=tf.float32)
    h_bool = tf.cast(tf.greater(h, 0), dtype=tf.float32)
    ua = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1) + box_area - w * h
    return tf.nn.relu(w_bool * h_bool * tf.cast(w * h / ua, dtype=tf.float32))
  return tf.transpose(tf.map_fn(
    _compute_overlap,
    elems=[query_boxes],
    dtype=tf.float32
  ))

def annotations(anchors, bboxes, negative_overlap=0.4, positive_overlap=0.5):
  overlaps = compute_overlap(anchors, bboxes)
  indices = tf.argmax(overlaps, axis=1)
  slices = tf.stack([tf.range(overlaps.shape[0], dtype=tf.int64), indices], axis=1)
  max_overlaps = tf.gather_nd(overlaps, slices)
  positive_indices = max_overlaps >= positive_overlap
  ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices
  return positive_indices, ignore_indices, indices

def bbox_transform(anchors, gt_boxes, mean=None, std=None):
  # DOTO: optimize performance
  gt_boxes = tf.cast(gt_boxes, dtype=tf.float32)
  if mean is None:
    mean = tf.constant([0, 0, 0, 0], dtype=tf.float32)
  if std is None:
    std = tf.constant([0.2, 0.2, 0.2, 0.2], dtype=tf.float32)
  anchor_widths  = anchors[:, 2] - anchors[:, 0]
  anchor_heights = anchors[:, 3] - anchors[:, 1]
  targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
  targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
  targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
  targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights
  targets = tf.stack([targets_dx1, targets_dy1, targets_dx2, targets_dy2])
  targets = (tf.transpose(targets) - mean) / std
  return targets

def anchors_based_targets(anchors, images, bboxes, labels, num_classes, negative_overlap=0.4, positive_overlap=0.5):
  def _ignore_annotations(image, k):
    centers = tf.transpose(tf.stack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]))
    indices = tf.logical_or(centers[:, 0] >= image.shape[1], centers[:, 1] >= image.shape[0])
    return k * tf.cast(tf.logical_not(indices), dtype=tf.float32) - tf.cast(indices, dtype=tf.float32)
  def _anchors(arguments):
    image, bbox, label = arguments[0:3]
    if bbox.shape[0]:
      positive_indices, ignore_indices, argmax_overlaps_indeces = annotations(
        anchors, bbox, negative_overlap=negative_overlap, positive_overlap=positive_overlap
      )
      k = (tf.zeros(shape=(anchors.shape[0]), dtype=tf.float32) -
            tf.cast(ignore_indices, dtype=tf.float32) + tf.cast(positive_indices, dtype=tf.float32))
      if image.shape:
        k = _ignore_annotations(image, k)
      k = tf.reshape(k, (-1, 1))
      localization = tf.concat([bbox_transform(anchors, tf.gather(bbox, argmax_overlaps_indeces, axis=0)), k], axis=1)
      label = tf.cast(label, dtype=tf.int32)
      classification = tf.concat([
        tf.reshape(tf.transpose(tf.one_hot(tf.gather(label, argmax_overlaps_indeces, axis=0), depth=num_classes)) *
          tf.cast(positive_indices, dtype=tf.float32), (-1, 1)), k], axis=1)
      return localization, classification
    else:
      k = tf.zeros(shape=(anchors.shape[0]), dtype=tf.float32)
      localization = tf.zeros(shape=(anchors.shape[0], 4), dtype=tf.float32)
      classification = tf.zeros(shape=(anchors.shape[0], num_classes), dtype=tf.float32)
      if image.shape:
        k = _ignore_annotations(image, k)
      k = tf.reshape(k, (-1, 1))
      localization = tf.concat([localization, k], axis=1)
      classification = tf.concat([classification, k], axis=1)
      return localization, classification
  return tf.map_fn(
    _anchors,
    elems=[images, bboxes, labels],
    dtype=(tf.float32, tf.float32)
  )

def build(base_size, ratios, scales):
  shapes = tf.constant([ratios.shape[0], scales.shape[0]])
  scales = tf.reshape(base_size * tf.tile(tf.expand_dims(scales, axis=0), (1, shapes[0])), (-1, ))
  ratios = tf.repeat(ratios, shapes[1])
  width  = tf.sqrt(scales * scales / ratios)
  height = width * ratios
  return tf.transpose(tf.stack([0 - width, 0 - height, width, height], axis=0)) * 0.5
