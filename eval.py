#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from operator import itemgetter

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow.keras.losses
import numpy as np

from dataset import Dataset
from metrics import weighted_bce_dice_loss, weighted_binary_crossentropy
from constant import *
import config_eval
from model import compile_model
from util import get_metrics, get_custom_objects

FLAGS = tf.flags.FLAGS

tensorflow.keras.losses.weighted_bce_dice_loss = weighted_bce_dice_loss


def eval(dataset):

    path_model = os.path.join(FLAGS.model, NAME_MODEL)

    config = tf.ConfigProto(
        allow_soft_placement=True,  gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.8, allow_growth=True))

    if FLAGS.weight_ad is not None:
        weight_adaptive = [float(x) for x in FLAGS.weight_ad]
    else:
        weight_adaptive = None

    with tf.Graph().as_default():
        iter_train, iter_valid = dataset.gen_train_valid(
            n_splits=N_SPLITS, idx_kfold=FLAGS.cv, batch_size=FLAGS.batch_size, adjust=FLAGS.adjust,
            weight_fg=FLAGS.weight_fg, weight_bg=FLAGS.weight_bg, weight_adaptive=weight_adaptive, repeat=1,
            filter_vert_hori=False)

        num_train, num_valid = dataset.len_train_valid(N_SPLITS, FLAGS.cv)

        with tf.Session(config=config) as sess:
            K.set_session(sess)
            steps_train = int(np.ceil(num_train / FLAGS.batch_size))
            steps_valid = int(np.ceil(num_valid / FLAGS.batch_size))

            model = load_model(path_model, compile=False)

            threshold = FLAGS.threshold
            if FLAGS.best_threshold:
                print("Searching best threshold for validation data")
                threshold = search_best_threshod(model, sess, iter_valid, steps_valid)
                print("Best threshold is {}".format(threshold))

            model = compile_model(model, optimizer="adam", loss=FLAGS.loss, threshold=threshold)

            sess.run([iter_train.initializer, iter_valid.initializer])
            metrics = model.evaluate(x=iter_train, steps=steps_train)
            print("Training loss:{}, iou:{}, score:{} (threshold={})".format(metrics[0], metrics[1], metrics[2], threshold))
            metrics = model.evaluate(x=iter_valid, steps=steps_valid)
            print("Validation loss:{}, iou:{}, score:{} (threshold={})".format(metrics[0], metrics[1], metrics[2], threshold))

def search_best_threshod(model, sess, iterator, steps):
    # _model = deepcopy(model)
    scores = {}
    for threshold in np.arange(0.0, 1.0001, 0.05):
        sess.run(iterator.initializer)
        model.compile(optimizer="adam", loss=weighted_bce_dice_loss, metrics=get_metrics(threshold))
        metrics = model.evaluate(x=iterator, steps=steps)
        scores[threshold] = metrics[2]

    print(scores)
    threshold_best = max(scores.items(), key=itemgetter(1))[0]
    return threshold_best


def main(argv=None):
    dataset = Dataset(FLAGS.input)
    eval(dataset)


if __name__ == '__main__':
    tf.app.run()