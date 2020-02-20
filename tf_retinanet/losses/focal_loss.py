import tensorflow as tf

def focal(y_true, y_pred, alpha=0.25, gamma=2.0):
  labels = y_true[:, :, :-1]
  anchor_state = y_true[:, :, -1]
  classification = y_pred

  indices = tf.where(tf.keras.backend.not_equal(anchor_state, -1))
  labels = tf.gather_nd(labels, indices)
  classification = tf.gather_nd(classification, indices)

  alpha_factor = tf.ones_like(labels) * alpha
  alpha_factor = tf.where(tf.equal(labels, 1), alpha_factor, 1 - alpha_factor)
  focal_weight = tf.where(tf.equal(labels, 1), 1 - classification, classification)
  focal_weight = alpha_factor * focal_weight ** gamma

  cls_loss = focal_weight * tf.keras.losses.binary_crossentropy(labels, classification)

  normalizer = tf.where(tf.equal(anchor_state, 1))
  normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
  normalizer = tf.math.maximum(1.0, normalizer)

  return tf.math.reduce_sum(cls_loss) / normalizer
