#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as K

from model import build_model
from input import Dataset
from constant import *
from util import StepDecay, MyTensorBoard
import train_config

FLAGS = tf.flags.FLAGS


def augment_dict():
    return dict(
        horizontal_flip=FLAGS.horizontal_flip,
        vertical_flip=FLAGS.vertical_flip,
        rotation_range=FLAGS.rotation_range,
        zoom_range=FLAGS.zoom_range,
        width_shift_range=FLAGS.shift_range,
        height_shift_range=FLAGS.shift_range,
        fill_mode=FLAGS.fill_mode)


def train(dataset):
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,  gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.9, allow_growth=True)))
    K.set_session(sess)
    with tf.device('/gpu:0'):
        model = build_model(IM_HEIGHT, IM_WIDTH, IM_CHAN, batch_norm=FLAGS.batch_norm, drop_out=FLAGS.drop_out, dice=FLAGS.dice)

    print(model.summary())

    path_model = os.path.join(FLAGS.model, NAME_MODEL)

    checkpointer = ModelCheckpoint(path_model, monitor='val_mean_score', verbose=1, save_best_only=True, mode='max')
    tensorboarder = MyTensorBoard(FLAGS.log, model=model)
    lrscheduler = LearningRateScheduler(StepDecay(FLAGS.lr, FLAGS.lr_decay, FLAGS.epochs_decay), verbose=1)

    callbacks = [checkpointer, tensorboarder, lrscheduler]
    if FLAGS.early_stopping:
        callbacks += EarlyStopping(patience=5, verbose=1)

    if FLAGS.legacy:
        VALIDATION_SPLIT = 0.1
        results = model.fit(
            dataset.X_samples, dataset.Y_samples, validation_split=VALIDATION_SPLIT,
            batch_size=FLAGS.batch_size, epochs=FLAGS.epochs,
            callbacks=callbacks)
    else:
        train_generator, valid_generator = dataset.create_generator_cv(
            n_splits=N_SPLITS, idx_kfold=FLAGS.cv, batch_size=FLAGS.batch_size, augment_dict=augment_dict())
        steps_per_epoch = int(dataset.num_train / FLAGS.batch_size)
        validation_steps = int(dataset.num_valid / FLAGS.batch_size)

        results = model.fit_generator(
            generator=train_generator, validation_data=valid_generator,
            epochs=FLAGS.epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
            shuffle=True, max_queue_size=steps_per_epoch, workers=INPUT_WORKERS,
            callbacks=callbacks)


def main(argv=None):
    dataset = Dataset(FLAGS.input)

    if tf.gfile.Exists(FLAGS.model):
        tf.gfile.DeleteRecursively(FLAGS.model)
    tf.gfile.MakeDirs(FLAGS.model)
    if tf.gfile.Exists(FLAGS.log):
        tf.gfile.DeleteRecursively(FLAGS.log)
    tf.gfile.MakeDirs(FLAGS.log)

    train(dataset)

if __name__ == '__main__':
    tf.app.run()