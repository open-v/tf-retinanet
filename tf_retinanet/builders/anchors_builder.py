import numpy as np
import tensorflow as tf
from tf_retinanet.builders import config_builder
from tf_retinanet.builders.compute_overlap import compute_overlap

def guess_shapes(image_shape, pyramid_levels):
  image_shape = np.array(image_shape[:2])
  image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
  return image_shapes

def shift(shape, stride, anchors):
  shift_x = (np.arange(0, shape[1]) + 0.5) * stride
  shift_y = (np.arange(0, shape[0]) + 0.5) * stride
  shift_x, shift_y = np.meshgrid(shift_x, shift_y)
  shifts = np.vstack((
    shift_x.ravel(), shift_y.ravel(),
    shift_x.ravel(), shift_y.ravel()
  )).transpose()
  A = anchors.shape[0]
  K = shifts.shape[0]
  anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
  anchors = anchors.reshape((K * A, 4))
  return anchors

def generate_anchors(base_size, ratios, scales):
  num_anchors = len(ratios) * len(scales)
  anchors = np.zeros((num_anchors, 4))
  anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
  areas = anchors[:, 2] * anchors[:, 3]
  anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
  anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
  anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
  anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
  return anchors

def anchors_for_shape(image_shape, anchor_params=None, pyramid_levels=None):
  if anchor_params is None:
    anchor_params = config_builder.build_anchors()
  if pyramid_levels is None:
    pyramid_levels = [3, 4, 5, 6, 7]
  image_shapes = guess_shapes(image_shape, pyramid_levels)
  anchors = np.zeros((0, 4))
  for index, pyramid in enumerate(pyramid_levels):
    _anchors = generate_anchors(
      base_size=anchor_params['sizes'][index],
      ratios=anchor_params['ratios'],
      scales=anchor_params['scales']
    )
    shifted_anchors = shift(image_shapes[index], anchor_params['strides'][index], _anchors)
    anchors = np.append(anchors, shifted_anchors, axis=0)
  return anchors

def anchor_targets_bbox(anchors, images, bboxes, labels, num_classes, negative_overlap=0.4,
    positive_overlap=0.5):
  batch_size = images.shape[0]
  localization_batch = np.zeros((batch_size, anchors.shape[0], 4 + 1),
    dtype=tf.keras.backend.floatx())
  classification_batch = np.zeros((batch_size, anchors.shape[0], num_classes + 1),
    dtype=tf.keras.backend.floatx())
  for index in range(images.shape[0]):
    image, bbox, label = images[index], bboxes[index], labels[index]
    if bbox.shape[0]:
      positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(
        anchors, bbox, negative_overlap, positive_overlap
      )
      classification_batch[index, ignore_indices, -1] = -1
      classification_batch[index, positive_indices, -1] = 1
      localization_batch[index, ignore_indices, -1] = -1
      localization_batch[index, positive_indices, -1] = 1
      classification_batch[index, positive_indices, label[argmax_overlaps_inds].astype(int)] = 1
      localization_batch[index, :, :-1] = bbox_transform(anchors, bbox[argmax_overlaps_inds, :])
    if image.shape:
      anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
      indices = np.logical_or(anchors_centers[:, 0] >= image.shape[1], anchors_centers[:, 1] >= image.shape[0])
      classification_batch[index, indices, -1] = -1
      localization_batch[index, indices, -1] = -1
  return localization_batch, classification_batch

def compute_gt_annotations(anchors, bboxes, negative_overlap=0.4, positive_overlap=0.5):
  overlaps = compute_overlap(anchors.astype(np.float64), bboxes.astype(np.float64))
  argmax_overlaps_inds = np.argmax(overlaps, axis=1)
  max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]
  positive_indices = max_overlaps >= positive_overlap
  ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices
  return positive_indices, ignore_indices, argmax_overlaps_inds

def bbox_transform(anchors, gt_boxes, mean=None, std=None):
  if mean is None:
    mean = np.array([0, 0, 0, 0])
  if std is None:
    std = np.array([0.2, 0.2, 0.2, 0.2])
  if isinstance(mean, (list, tuple)):
    mean = np.array(mean)
  if isinstance(std, (list, tuple)):
    std = np.array(std)
  anchor_widths  = anchors[:, 2] - anchors[:, 0]
  anchor_heights = anchors[:, 3] - anchors[:, 1]

  targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
  targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
  targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
  targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

  targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
  targets = targets.T

  targets = (targets - mean) / std
  return targets
