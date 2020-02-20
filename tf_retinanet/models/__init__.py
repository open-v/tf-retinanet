import tf_retinanet
import tensorflow as tf
from .feature_extractor import FeatureExtractor
from .classification_feature_extractor import ClassificationFeatureExtractor
from .localization_feature_extractor import LocalizationFeatureExtractor
from .pyramid_feature_extractor import PyramidFeatureExtractor

class RetinaNet(tf.keras.Model):
  def __init__(self, backbone, num_classes=None, num_values=None, num_anchors=None,
        name='retinanet', anchors_sizes=None, anchors_strides=None,
        anchors_ratios=None, anchors_scales=None, **kwargs):
    super(RetinaNet, self).__init__(**kwargs)

    if num_classes is None:
      num_classes = tf_retinanet.configs.config.num_classes
    if num_values is None:
      num_values = tf_retinanet.configs.config.num_localization_values
    if num_anchors is None:
      num_anchors = tf_retinanet.configs.config.anchors.num_anchors

    self.backbone = backbone
    self.pyramid = PyramidFeatureExtractor()
    self.classification = ClassificationFeatureExtractor(num_classes, num_anchors,
        name='{}_classification'.format(name))
    self.localization = LocalizationFeatureExtractor(num_values, num_anchors,
        name='{}_localization'.format(name))
    self.concatenate = tf.keras.layers.Concatenate(name='{}_concatenate'.format(name), axis=1)
    self.anchors = tf_retinanet.layers.Anchors(
      sizes=anchors_sizes,
      strides=anchors_strides,
      ratios=anchors_ratios,
      scales=anchors_scales
    )

  def call(self, image, detection=False, filter_detections=True, max_detections=300,
        iou_threshold=0.5, **kwargs):

    def transformed_anchors(anchors, deltas, mean=[0, 0, 0, 0], std=[0.2, 0.2, 0.2, 0.2]):
      width  = anchors[:, :, 2] - anchors[:, :, 0]
      height = anchors[:, :, 3] - anchors[:, :, 1]
      x1 = anchors[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
      y1 = anchors[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
      x2 = anchors[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
      y2 = anchors[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height
      return tf.stack([x1, y1, x2, y2], axis=2)

    def clipped_anchors(image, anchors):
      shape = tf.cast(tf.shape(image), dtype=tf.float32)
      _, height, width, _ = tf.unstack(shape, axis=0)
      x1, y1, x2, y2 = tf.unstack(anchors, axis=-1)
      x1 = tf.clip_by_value(x1, 0, width  - 1)
      y1 = tf.clip_by_value(y1, 0, height - 1)
      x2 = tf.clip_by_value(x2, 0, width  - 1)
      y2 = tf.clip_by_value(y2, 0, height - 1)
      return tf.stack([x1, y1, x2, y2], axis=2)

    def _filter_detections(args):
      boxes       = args[0]
      classes     = args[1]
      return self._filter_detections(boxes, classes,
          max_detections=max_detections, iou_threshold=iou_threshold)

    features         = self.backbone(image)
    pyramid_features = self.pyramid(features)
    localization     = self.concatenate([self.localization(feature) for feature in pyramid_features])
    classification   = self.concatenate([self.classification(feature) for feature in pyramid_features])
    if detection:
      anchors = self.anchors(pyramid_features)
      anchors = transformed_anchors(anchors, localization)
      anchors = clipped_anchors(image, anchors)
      if filter_detections:
        return tf.map_fn(
          _filter_detections,
          elems=[anchors, classification],
          dtype=[tf.float32, tf.float32, tf.int32],
          parallel_iterations=1
        )
      return [anchors, classification]
    return [localization, classification]

  def _filter_detections(self, boxes, classes, max_detections=300, iou_threshold=0.5):
    all_indices = []
    for c in range(int(classes.shape[1])):
      scores = classes[:, c]
      labels = c * tf.ones((scores.shape[0],), dtype=tf.int64)
      indices = tf.where(tf.greater(scores, 0.05))
      filtered_boxes = tf.gather_nd(boxes, indices)
      filtered_scores = tf.gather(scores, indices)[:, 0]
      nms_indices = tf.image.non_max_suppression(
        filtered_boxes, filtered_scores, max_output_size=300, iou_threshold=0.5)
      indices = tf.gather(indices, nms_indices)
      labels = tf.gather_nd(labels, indices)
      indices = tf.stack([indices[:, 0], labels], axis=1)
      all_indices.append(indices)

    indices = tf.concat(all_indices, axis=0)
    scores = tf.gather_nd(classes, indices)
    labels = indices[:, 1]

    scores, top_indices = tf.nn.top_k(scores, k=tf.math.minimum(300, tf.shape(scores)[0]))
    indices = tf.gather(indices[:, 0], top_indices)
    boxes = tf.gather(boxes, indices)
    labels = tf.gather(labels, top_indices)

    pad_size = tf.math.maximum(0, 300 - tf.shape(scores)[0])

    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels = tf.cast(labels, tf.int32)
    return [boxes, scores, labels]
