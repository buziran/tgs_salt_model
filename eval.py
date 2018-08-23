#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow.keras.losses

from input import input_train, input_test
from metrics import mean_iou, mean_score, bce_dice_loss
from config import *

tf.flags.DEFINE_string(
    'input', None,
    """path to train data""")

tf.flags.DEFINE_string(
    'model', 'output/model',
    """path to model directory""")

tf.flags.DEFINE_bool('dice', False, """whether to use dice loss""")

FLAGS = tf.flags.FLAGS

tensorflow.keras.losses.bce_dice_loss = bce_dice_loss


def eval(dir_train):

    X_train, Y_train = input_train(dir_train)
    path_model = os.path.join(FLAGS.model, name_model)

    config = tf.ConfigProto(
        allow_soft_placement=True,  gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.9, allow_growth=True))

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            K.set_session(sess)
            model = load_model(path_model, custom_objects={'mean_iou': mean_iou, 'mean_score': mean_score}, compile=False)
            if FLAGS.dice:
                loss = bce_dice_loss
            else:
                loss = 'binary_crossentropy'
            model.compile(optimizer="adam", loss=loss, metrics=[mean_iou, mean_score])

            metrics = model.evaluate(
                X_train[:int(X_train.shape[0] * 0.9)], Y_train[:int(X_train.shape[0] * 0.9)], batch_size=8, verbose=1)
            print("Training loss:{}, iou:{}, score:{}".format(metrics[0], metrics[1], metrics[2]))
            metrics = model.evaluate(
                X_train[int(X_train.shape[0] * 0.9):], Y_train[int(X_train.shape[0] * 0.9):], batch_size=8, verbose=1)
            print("Validation loss:{}, iou:{}, score:{}".format(metrics[0], metrics[1], metrics[2]))


def main(argv=None):
    eval(FLAGS.input)


if __name__ == '__main__':
    tf.app.run()