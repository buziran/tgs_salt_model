#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from input import input_train, input_test
from metrics import mean_iou, mean_score
from config import *

tf.flags.DEFINE_string(
    'input_train', None,
    """path to train data""")

tf.flags.DEFINE_string(
    'input_test', None,
    """path to test data""")

tf.flags.DEFINE_string(
    'model', 'output/model',
    """path to model directory""")

FLAGS = tf.flags.FLAGS


def eval(dir_train):

    X_train, Y_train = input_train(dir_train)
    path_model = os.path.join(FLAGS.model, name_model)

    config = tf.ConfigProto(
        allow_soft_placement=True,  gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.9, allow_growth=True))

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            K.set_session(sess)
            model = load_model(path_model, custom_objects={'mean_iou': mean_iou, 'mean_score': mean_score})

            metrics = model.evaluate(
                X_train[:int(X_train.shape[0] * 0.9)], Y_train[:int(X_train.shape[0] * 0.9)], batch_size=8, verbose=1)
            print("Training loss:{}, iou:{}, score:{}".format(metrics[0], metrics[1], metrics[2]))
            metrics = model.evaluate(
                X_train[int(X_train.shape[0] * 0.9):], Y_train[int(X_train.shape[0] * 0.9):], batch_size=8, verbose=1)
            print("Validation loss:{}, iou:{}, score:{}".format(metrics[0], metrics[1], metrics[2]))


def main(argv=None):
    eval(FLAGS.input_train)


if __name__ == '__main__':
    tf.app.run()