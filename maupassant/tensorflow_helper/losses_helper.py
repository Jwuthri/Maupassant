import tensorflow as tf
import tensorflow.keras.backend as K

e = K.epsilon()


@tf.function
def f1_loss(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels)."""
    y_hat = tf.cast(y_hat, tf.float32)
    y = tf.cast(y, tf.float32)

    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + e)
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
    iou = (intersection + e) / (union + e)

    return 1 - iou


@tf.function
def focal_loss(y, y_hat, alpha=0.8, gamma=2):
    """Focal Loss was introduced by Lin et al of Facebook AI Research in 2017 as a means of combatting extremely
    imbalanced datasets where positive cases were relatively rare:
    https://arxiv.org/abs/1708.02002"""
    y_hat = K.flatten(y_hat)
    y = K.flatten(y)

    bce = K.binary_crossentropy(y, y_hat)
    bce_exp = K.exp(-bce)
    focal = K.mean(alpha * K.pow((1 - bce_exp), gamma) * bce)

    return focal


@tf.function
def focal_loss_fixed(y_true, y_pred, gamma=2, alpha=0.4):
    """Focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: y_pred is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002
    """
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred, e)
    ce = tf.multiply(y_true, -tf.math.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)

    return tf.reduce_mean(reduced_fl)


@tf.function
def combo_loss(y, y_hat, alpha=0.5, ce_ratio=0.5, smooth=1):
    """Combo loss is a combination of Dice Loss and a modified Cross-Entropy function that, like Tversky loss, has
    additional constants which penalise either false positives or false negatives more respectively:
    https://arxiv.org/abs/1805.02798"""
    y = K.flatten(y)
    y_hat = K.flatten(y_hat)

    intersection = K.sum(y * y_hat)
    dice = (2. * intersection + smooth) / (K.sum(y) + K.sum(y_hat) + smooth)
    y_hat = K.clip(y_hat, e, 1.0 - e)
    out = - (alpha * ((y * K.log(y_hat)) + ((1 - alpha) * (1.0 - y) * K.log(1.0 - y_hat))))
    weighted_ce = K.mean(out, axis=-1)
    combo = (ce_ratio * weighted_ce) - ((1 - ce_ratio) * dice)

    return combo
