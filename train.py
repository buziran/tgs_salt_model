#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow.keras.backend as K

from model import build_model
from input import input_train, create_generator
from config import *

tf.flags.DEFINE_string(
    'input', None, """path to train data""")

tf.flags.DEFINE_string(
    'model', 'output/model', """path to model directory""")

tf.flags.DEFINE_string(
    'log', 'output/log', """path to log directory""")

tf.flags.DEFINE_integer(
    'epochs', 30, """path to log directory""")

tf.flags.DEFINE_integer(
    'batch_size', 8, """batch size""")

tf.flags.DEFINE_bool(
    'augment', True, """whether to apply augmentation""")

FLAGS = tf.flags.FLAGS

BATCH_SIZE = 8
VALIDATION_SPLIT = 0.1
INPUT_WORKERS = 4


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
    if not FLAGS.augment:
        results = model.fit(
            X_train, Y_train, validation_split=VALIDATION_SPLIT, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs,
            callbacks=[earlystopper, checkpointer, tensorboarder])
    else:
        num_train_image = int(X_train.shape[0] * (1. - VALIDATION_SPLIT))
        num_valid_image = int(X_train.shape[0] * VALIDATION_SPLIT)
        steps_per_epoch = int(num_train_image / FLAGS.batch_size)
        validation_steps = int(num_valid_image / FLAGS.batch_size)
        train_generator, valid_generator = create_generator(
            X_train, Y_train, batch_size=FLAGS.batch_size, validation_split=VALIDATION_SPLIT)

        # import numpy as np
        # for e in range(FLAGS.epochs):
        #     for i, (x, y) in enumerate(train_generator):
        #         if i == 0:
        #             print(np.sum(y))
        #         if i == steps_per_epoch - 1:
        #             break

        results = model.fit_generator(
            generator=train_generator, validation_data=valid_generator,
            epochs=FLAGS.epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
            shuffle=True, max_queue_size=steps_per_epoch, workers=INPUT_WORKERS,
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