import tensorflow as tf

def smooth_l1(y_true, y_pred, sigma=3.0):
  sigma_squared = sigma ** 2
  localization = y_pred
  localization_target = y_true[:, :, :-1]
  anchor_state = y_true[:, :, -1]

  indices = tf.where(tf.equal(anchor_state, 1))
  localization = tf.gather_nd(localization, indices)
  localization_target = tf.gather_nd(localization_target, indices)

  localization_diff = localization - localization_target
  localization_diff = tf.math.abs(localization_diff)
  localization_loss = tf.where(
    tf.math.less(localization_diff, 1.0 / sigma_squared),
    0.5 * sigma_squared * tf.math.pow(localization_diff, 2),
    localization_diff - 0.5 / sigma_squared
  )

  normalizer = tf.math.maximum(1, tf.shape(indices)[0])
  normalizer = tf.cast(normalizer, dtype=tf.float32)
  return tf.math.reduce_sum(localization_loss) / normalizer
