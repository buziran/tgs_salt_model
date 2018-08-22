#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler
import tensorflow.keras.backend as K

from model import build_model
from input import Dataset
from config import *
from util import StepDecay, MyTensorBoard

tf.flags.DEFINE_string(
    'input', None, """path to train data""")

tf.flags.DEFINE_string(
    'model', 'output/model', """path to model directory""")

tf.flags.DEFINE_string(
    'log', 'output/log', """path to log directory""")

tf.flags.DEFINE_integer(
    'epochs', 100, """path to log directory""")

tf.flags.DEFINE_integer(
    'batch_size', 8, """batch size""")

tf.flags.DEFINE_integer(
    'idx_kfold', 0, help="""index of k-fold cross validation. index must be in 0~9""")

tf.flags.DEFINE_bool(
    'early_stopping', False, help="""whether to apply early-stopping""")

tf.flags.DEFINE_float(
    'lr', 0.001, help="""initial value of learning rate""")

tf.flags.DEFINE_float(
    'lr_decay', 1.0, help="""decay factor for learning rate""")

tf.flags.DEFINE_integer(
    'epochs_decay', 10, help="""decay epoch of learning rate""")

"""Augmentations"""
tf.flags.DEFINE_bool(
    'legacy', False, """whether to use legacy code""")

tf.flags.DEFINE_bool(
    'horizontal_flip', False, """whether to apply horizontal flip""")

tf.flags.DEFINE_bool(
    'vertical_flip', False, """whether to apply vertical flip""")

tf.flags.DEFINE_integer(
    'rotation_range', 0, """random rotation range""")

tf.flags.DEFINE_float(
    'zoom_range', 0., """random zoom range""")

tf.flags.DEFINE_enum(
    'fill_mode', 'reflect', enum_values=['constant', 'nearest', 'reflect', 'wrap'], help="""fill mode""")

FLAGS = tf.flags.FLAGS

N_SPLITS = 10
BATCH_SIZE = 8
INPUT_WORKERS = 4


def augment_dict():
    return dict(
        horizontal_flip=FLAGS.horizontal_flip,
        vertical_flip=FLAGS.vertical_flip,
        rotation_range=FLAGS.rotation_range,
        zoom_range=FLAGS.zoom_range,
        fill_mode=FLAGS.fill_mode)


def train(dataset):
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,  gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.9, allow_growth=True)))
    K.set_session(sess)
    with tf.device('/gpu:0'):
        model = build_model(im_height, im_width, im_chan)

    path_model = os.path.join(FLAGS.model, name_model)

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
        train_generator, valid_generator = dataset.create_generator(
            n_splits=N_SPLITS, idx_kfold=FLAGS.idx_kfold, batch_size=FLAGS.batch_size, augment_dict=augment_dict())
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