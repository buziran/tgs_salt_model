import tensorflow as tf
from tensorflow.keras import backend as K
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


def mean_score(y_true, y_pred):
    y_true_ = tf.cast(y_true, tf.bool)
    y_pred_ = tf.cast(tf.round(y_pred), tf.bool)

    # 画像ごとにflatten
    y_true_ = tf.reshape(y_true_, shape=[tf.shape(y_true_)[0], -1])
    y_pred_ = tf.reshape(y_pred_, shape=[tf.shape(y_pred_)[0], -1])
    threasholds_iou = tf.constant(np.arange(0.5, 1.0, 0.05), dtype=tf.float32)

    def _mean_score(y):
        y0, y1 = y[0], y[1]
        # y0 = tf.Print(y0, [tf.shape(y0), tf.shape(y1)], message="shapes ")
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
