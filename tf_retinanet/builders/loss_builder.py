import tensorflow as tf

def build_smooth_l1_loss(sigma=3.0):
  sigma_squared = sigma ** 2
  def smooth_l1(y_true, y_pred):
    localization = y_pred
    localization_target = y_true[:, :, :-1]
    anchor_state = y_true[:, :, -1]

    indices = tf.where(tf.keras.backend.equal(anchor_state, 1))
    localization = tf.gather_nd(regression, indices)
    localization_target = tf.gather_nd(regression_target, indices)

    localization_diff = regression - regression_target
    localization_diff = tf.keras.backend.abs(regression_diff)
    localization_abs = tf.where(
      tf.keras.backend.less(regression_diff, 1.0 / sigma_squared),
      0.5 * sigma_squared * tf.keras.backend.pow(regression_diff, 2),
      regression_diff - 0.5 / sigma_squared
    )

    normalizer = tf.keras.backend.maximum(1, tf.keras.backend.shape(indices)[0])
    normalizer = tf.keras.backend.cast(normalizer, dtype=tf.keras.backend.floatx())
    return tf.keras.backend.sum(regression_loss) / normalizer
  return smooth_l1

def build_focal_loss(alpha=0.25, gamma=2.0):
  def focal(y_true, y_pred):
    labels = y_true[:, :, :-1]
    anchor_state = y_true[:, :, -1]
    classification = y_pred

    indices = tf.where(tf.keras.backend.not_equal(anchor_state, -1))
    labels = tf.gather_nd(labels, indices)
    classification = tf.gather_nd(classification, indices)

    alpha_factor = tf.keras.backend.ones_like(labels) * alpha
    alpha_factor = tf.where(tf.keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(tf.keras.backend.equal(labels, 1), 1 - classification, classification)
    focal_weight = alpha_factor * focal_weight ** gamma

    cls_loss = focal_weight * tf.keras.backend.binary_crossentropy(labels, classification)

    normalizer = tf.where(tf.keras.backend.equal(anchor_state, 1))
    normalizer = tf.keras.backend.cast(tf.keras.backend.shape(normalizer)[0], tf.keras.backend.floatx())
    normalizer = tf.keras.backend.maximum(tf.keras.backend.cast_to_floatx(1.0), normalizer)

    return tf.keras.backend.sum(cls_loss) / normalizer
  return focal

losses = {
  'smooth_l1': build_smooth_l1_loss,
  'focal': build_focal_loss,
}
def build(name, **kwargs):
  if name in losses:
    return losses[name](**kwargs)
  else:
    raise ValueError('Loss ({}) is not supported'.format(name))
