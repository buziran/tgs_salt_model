import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
import numpy as np


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def weighted_mean_iou(y_true_and_weight, y_pred):
    y_true, weight = split_label_weight(y_true_and_weight)
    mask = tf.greater(weight, 0)
    return mean_iou(y_true * tf.cast(mask, y_true.dtype), y_pred * tf.cast(mask, y_true.dtype))


def mean_score(y_true, y_pred, threshold=None):
    """
    Calculate mean score for batch images

    :param y_true: 4-D Tensor of ground truth, such as [NHWC]. Should have numeric or boolean type.
    :param y_pred: 4-D Tensor of prediction, such as [NHWC]. Should have numeric or boolean type.
    :return: 0-D Tensor of score
    """
    y_true_ = tf.cast(tf.round(y_true), tf.bool)
    if threshold is None:
        y_pred_ = tf.cast(tf.round(y_pred), tf.bool)
    else:
        y_pred_ = tf.greater(y_pred, threshold)

    # 画像ごとにflatten
    y_true_ = tf.reshape(y_true_, shape=[tf.shape(y_true_)[0], -1])
    y_pred_ = tf.reshape(y_pred_, shape=[tf.shape(y_pred_)[0], -1])
    threasholds_iou = tf.constant(np.arange(0.5, 1.0, 0.05), dtype=tf.float32)

    def _mean_score(y):
        """Calculate score per image"""
        y0, y1 = y[0], y[1]
        total_cm = tf.confusion_matrix(y0, y1, num_classes=2)
        sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
        sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
        cm_diag = tf.to_float(tf.diag_part(total_cm))
        denominator = sum_over_row + sum_over_col - cm_diag
        denominator = tf.where(tf.greater(denominator, 0), denominator, tf.ones_like(denominator))
        # iou[0]: 背景のIoU
        # iou[1]: 前景のIoU
        iou = tf.div(cm_diag, denominator)
        iou_fg = iou[1]
        greater = tf.greater(iou_fg, threasholds_iou)
        score_per_image = tf.reduce_mean(tf.cast(greater, tf.float32))
        # GT, Predともに前景ゼロの場合はスコアを1とする
        score_per_image = tf.where(
            tf.logical_and(
                tf.equal(tf.reduce_any(y0), False), tf.equal(tf.reduce_any(y1), False)),
            1., score_per_image)
        return score_per_image

    elems = (y_true_, y_pred_)
    scores_per_image = tf.map_fn(_mean_score, elems, dtype=tf.float32)
    return tf.reduce_mean(scores_per_image)


def weighted_mean_score(y_true_and_weight, y_pred, threshold=None):
    y_true, weight = split_label_weight(y_true_and_weight)
    mask = tf.to_int32(tf.greater(weight, 0))
    return mean_score(y_true * tf.cast(mask, y_true.dtype), y_pred * tf.cast(mask, y_true.dtype), threshold)


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def weighted_binary_crossentropy(y_true_and_weight, y_pred):
    y_true, weight = tf.split(y_true_and_weight, [1, 1], axis=3)
    bce = K.binary_crossentropy(y_true, y_pred)
    wbce = bce * weight
    return K.mean(wbce)


def weighted_bce_dice_loss(y_true_and_weight, y_pred):
    y_true, weight = split_label_weight(y_true_and_weight)
    mask = tf.greater(weight, 1)
    wbce = weighted_binary_crossentropy(y_true_and_weight, y_pred)
    dloss = dice_loss(y_true * tf.cast(mask, y_true.dtype), y_pred * tf.cast(mask, y_true.dtype))

    return wbce + dloss


def mean_score_per_image(y_true, y_pred, threshold=None):
    """Calculate score per image"""
    # GT, Predともに前景ゼロの場合はスコアを1とする
    y_true = np.round(y_true).astype(np.int)
    if threshold is None:
        y_pred = np.round(y_pred).astype(np.int)
    else:
        y_pred = (y_pred>threshold).astype(np.int)

    if np.any(y_true) == False and np.any(y_pred) == False:
        return 1.

    threasholds_iou = np.arange(0.5, 1.0, 0.05, dtype=float)
    y_true = np.reshape(y_true, (-1))
    y_pred = np.reshape(y_pred, (-1))
    total_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sum_over_row = np.sum(total_cm, 0).astype(float)
    sum_over_col = np.sum(total_cm, 1).astype(float)
    cm_diag = np.diag(total_cm).astype(float)
    denominator = sum_over_row + sum_over_col - cm_diag
    denominator = np.where(np.greater(denominator, 0), denominator, np.ones_like(denominator))
    # iou[0]: 背景のIoU
    # iou[1]: 前景のIoU
    iou = np.divide(cm_diag, denominator)
    iou_fg = iou[1]
    greater = np.greater(iou_fg, threasholds_iou)
    score_per_image = np.mean(greater.astype(float))
    return score_per_image


def split_label_weight(label_and_weight):
    label, weight = tf.split(label_and_weight, [1, 1], axis=3)
    return label, weight

def l2_loss(weight_decay, exclude_bn):
    if exclude_bn:
        def _filter(name):
            return 'batch_normalization' not in name
    else:
        def _filter(name):
            return True
    loss_filter_fn = _filter

    with tf.name_scope("l2_loss"):
        _l2_loss = weight_decay * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
             if loss_filter_fn(v.name)])
    tf.summary.scalar('l2_loss', _l2_loss)
    return _l2_loss

