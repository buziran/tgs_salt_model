#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

from input import input_train, input_test
from metrics import mean_iou
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


def main(argv=None):
    X_train, Y_train = input_train(FLAGS.input_train)
    X_test = input_test(FLAGS.input_test)
    path_model = os.path.join(FLAGS.model, name_model)
    model = load_model(path_model, custom_objects={'mean_iou': mean_iou})
    preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
    preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
    preds_test = model.predict(X_test, verbose=1)

    # Threshold predictions
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    preds_test_t = (preds_test > 0.5).astype(np.uint8)

    ix = np.random.randint(0, len(preds_train_t))
    plt.imshow(np.dstack((X_train[ix], X_train[ix], X_train[ix])))
    plt.show()
    tmp = np.squeeze(Y_train[ix]).astype(np.float32)
    plt.imshow(np.dstack((tmp, tmp, tmp)))
    plt.show()
    tmp = np.squeeze(preds_train_t[ix]).astype(np.float32)
    plt.imshow(np.dstack((tmp, tmp, tmp)))
    plt.show()

if __name__ == '__main__':
    tf.app.run()