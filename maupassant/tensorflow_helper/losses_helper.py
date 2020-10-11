import tensorflow as tf
from tensorflow.python.ops import nn, math_ops
from tensorflow.python.framework import smart_cond, ops
from tensorflow.python.ops import array_ops

import tensorflow.keras.backend as K


@tf.function
def f1_loss(y, y_hat, label_smoothing=0.1):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels)."""
    y_hat = tf.cast(y_hat, tf.float32)
    y = tf.cast(y, tf.float32)
    label_smoothing = ops.convert_to_tensor_v2(label_smoothing, dtype=K.floatx())

    def _smooth_labels():
        num_classes = math_ops.cast(array_ops.shape(y)[-1], y_hat.dtype)
        return y * (1.0 - label_smoothing) + (label_smoothing / num_classes)

    y = smart_cond.smart_cond(label_smoothing, _smooth_labels, lambda: y)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + K.epsilon())
    cost = 1 - soft_f1
    macro_cost = tf.reduce_mean(cost)

    return macro_cost


@tf.function
def iou_loss(y, y_hat):
    """The IoU metric, or Jaccard Index, is similar to the Dice metric and is calculated as the ratio between the
    overlap of the positive instances between two sets, and their mutual combined values:
    https://arxiv.org/abs/1911.08287"""
    y_hat = K.flatten(y_hat)
    y = K.flatten(y)

    intersection = K.sum(K.dot(y, y_hat))
    total = K.sum(y) + K.sum(y_hat)
    union = total - intersection
    iou = (intersection + K.epsilon()) / (union + K.epsilon())

    return 1 - iou


@tf.function
def combo_loss(y, y_hat, alpha=0.5, ce_ratio=0.5, smooth=1):
    """Combo loss is a combination of Dice Loss and a modified Cross-Entropy function that, like Tversky loss, has
    additional constants which penalise either false positives or false negatives more respectively:
    https://arxiv.org/abs/1805.02798"""
    y = K.flatten(y)
    y_hat = K.flatten(y_hat)

    intersection = K.sum(y * y_hat)
    dice = (2. * intersection + smooth) / (K.sum(y) + K.sum(y_hat) + smooth)
    y_hat = K.clip(y_hat, K.epsilon(), 1.0 - K.epsilon())
    out = - (alpha * ((y * K.log(y_hat)) + ((1 - alpha) * (1.0 - y) * K.log(1.0 - y_hat))))
    weighted_ce = K.mean(out, axis=-1)
    combo = (ce_ratio * weighted_ce) - ((1 - ce_ratio) * dice)

    return combo


@tf.function
def cosine_similarity(y_true, y_pred, axis=-1):
    """Computes the cosine similarity between labels and predictions."""
    y_true = nn.l2_normalize(y_true, axis=axis)
    y_pred = nn.l2_normalize(y_pred, axis=axis)

    return -math_ops.reduce_sum(y_true * y_pred, axis=axis)


@tf.function
def focal_loss(y_true, y_pred, gamma=2.0, alpha=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    model_out = tf.add(y_pred, K.epsilon())
    ce = tf.multiply(y_true, -tf.math.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)

    return tf.reduce_mean(reduced_fl)
