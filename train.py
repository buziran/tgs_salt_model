#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow.keras.backend as K

from model import build_model
from input import input_train
from config import *

tf.flags.DEFINE_string(
    'input', None, """path to train data""")

tf.flags.DEFINE_string(
    'model', 'output/model', """path to model directory""")

tf.flags.DEFINE_string(
    'log', 'output/log', """path to log directory""")

tf.flags.DEFINE_integer(
    'epochs', 30, """path to log directory""")

FLAGS = tf.flags.FLAGS


def train(X_train, Y_train):
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,  gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.9, allow_growth=True)))
    K.set_session(sess)
    with tf.device('/gpu:0'):
        model = build_model(im_height, im_width, im_chan)

    path_model = os.path.join(FLAGS.model, name_model)

    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint(path_model, verbose=1, save_best_only=True)
    tensorboarder = TensorBoard(FLAGS.log)
    results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=FLAGS.epochs,
                        callbacks=[earlystopper, checkpointer, tensorboarder])


def main(argv=None):
    X_train, Y_train = input_train(FLAGS.input)

    if tf.gfile.Exists(FLAGS.model):
        tf.gfile.DeleteRecursively(FLAGS.model)
    tf.gfile.MakeDirs(FLAGS.model)
    if tf.gfile.Exists(FLAGS.log):
        tf.gfile.DeleteRecursively(FLAGS.log)
    tf.gfile.MakeDirs(FLAGS.log)

    train(X_train, Y_train)

if __name__ == '__main__':
    tf.app.run()