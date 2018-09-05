#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from pprint import pprint

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as K

from model import build_model, build_model_ref
from input import Dataset
from constant import *
from util import StepDecay, MyTensorBoard
import config_train

FLAGS = tf.flags.FLAGS

FLAGS_FILENAME = "flags.json"


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
    flag_values_dict = FLAGS.flag_values_dict()
    pprint(flag_values_dict, indent=4)
    with open(os.path.join(FLAGS.model, FLAGS_FILENAME), 'w') as f:
        json.dump(flag_values_dict, f, indent=4)

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,  gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.9, allow_growth=True)))
    K.set_session(sess)
    with tf.device('/gpu:0'):
        if not FLAGS.use_ref:
            model = build_model(
                IM_HEIGHT, IM_WIDTH, IM_CHAN, batch_norm=FLAGS.batch_norm, drop_out=FLAGS.drop_out, dice=FLAGS.dice)
        else:
            model = build_model_ref(
                IM_HEIGHT, IM_WIDTH, IM_CHAN, batch_norm=FLAGS.batch_norm, drop_out=FLAGS.drop_out, dice=FLAGS.dice,
                depth=FLAGS.depth, start_ch=FLAGS.start_ch)

    print(model.summary())

    path_model = os.path.join(FLAGS.model, NAME_MODEL)

    checkpointer = ModelCheckpoint(path_model, monitor='val_weighted_mean_score', verbose=1, save_best_only=True, mode='max')
    tensorboarder = MyTensorBoard(FLAGS.log, model=model)
    lrscheduler = LearningRateScheduler(StepDecay(FLAGS.lr, FLAGS.lr_decay, FLAGS.epochs_decay), verbose=1)

    callbacks = [checkpointer, tensorboarder, lrscheduler]
    if FLAGS.early_stopping:
        callbacks += EarlyStopping(patience=5, verbose=1)

    dataset.load_train(adjust=FLAGS.adjust)
    train_generator, valid_generator = dataset.create_train_generator(
        n_splits=N_SPLITS, idx_kfold=FLAGS.cv, batch_size=FLAGS.batch_size,
        augment_dict=augment_dict(), random_erase=FLAGS.random_erase)

    if FLAGS.debug:
        debug_img_show(train_generator, valid_generator)

    steps_per_epoch = int(dataset.num_train / FLAGS.batch_size)
    validation_steps = int(dataset.num_valid / FLAGS.batch_size)

    results = model.fit_generator(
        generator=train_generator, validation_data=valid_generator,
        epochs=FLAGS.epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
        shuffle=True, max_queue_size=steps_per_epoch, workers=INPUT_WORKERS,
        callbacks=callbacks)


def debug_img_show(train_generator, valid_generator):
    import numpy as np
    import matplotlib.pyplot as plt

    def show_img_label_mask(images, labels_and_masks, prefix=""):
        num_img = images.shape[0]
        for i, (image, label_and_mask) in enumerate(zip(images, labels_and_masks)):
            image = np.squeeze(image)
            label = label_and_mask[:,:,0]
            mask = label_and_mask[:,:,1]
            plt.title(prefix + "image {}/{}".format(i, num_img))
            plt.imshow(image, cmap='gray', vmin=0, vmax=255)
            plt.show()

            plt.title(prefix + "label {}/{}".format(i, num_img))
            plt.imshow(label, cmap='gray', vmin=0, vmax=1)
            plt.show()

            plt.title(prefix + "mask {}/{}".format(i, num_img))
            plt.imshow(mask, cmap='gray', vmin=0, vmax=np.max(mask))
            plt.colorbar()
            plt.show()

    for images, labels_and_masks in train_generator:
        show_img_label_mask(images, labels_and_masks, prefix="training ")
        break

    for images, labels_and_masks in valid_generator:
        show_img_label_mask(images, labels_and_masks, prefix="validation ")
        break

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