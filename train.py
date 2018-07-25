#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import build_model
from input import input_train
from config import *

tf.flags.DEFINE_string(
    'input', None,
    """path to train data""")

tf.flags.DEFINE_string(
    'input', None,
    """path to train data""")

tf.flags.DEFINE_string(
    'model', 'output/model',
    """path to model directory""")

FLAGS = tf.flags.FLAGS


def train(X_train, Y_train):
    with tf.device('/gpu:0'):
        model = build_model(im_height, im_width, im_chan)

    path_model = os.path.join(FLAGS.parameter, name_model)

    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint(path_model, verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30,
                        callbacks=[earlystopper, checkpointer])


def main(argv=None):
    X_train, Y_train = input_train(FLAGS.input)

    if tf.gfile.Exists(FLAGS.model):
        tf.gfile.DeleteRecursively(FLAGS.model)
    tf.gfile.MakeDirs(FLAGS.model)

    train(X_train, Y_train)

if __name__ == '__main__':
    tf.app.run()