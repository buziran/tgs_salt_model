#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from pprint import pprint

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model

from model import build_model, build_model_ref, build_model_pretrained, compile_model
from dataset import Dataset
from constant import *
from util import StepDecay, MyTensorBoard, write_summary
import config_train

FLAGS = tf.flags.FLAGS

FLAGS_FILENAME = "flags.json"
MODEL_SUMMARY_FILENAME = "model_summary.txt"


def augment_dict():
    if not FLAGS.augment:
        return None
    return dict(
        horizontal_flip=FLAGS.horizontal_flip,
        vertical_flip=FLAGS.vertical_flip,
        rotation_range=FLAGS.rotation_range,
        zoom_range=FLAGS.zoom_range,
        width_shift_range=FLAGS.shift_range,
        height_shift_range=FLAGS.shift_range,
        brightness_range=FLAGS.brightness_range,
        random_erase=FLAGS.random_erase,
        mixup=FLAGS.mixup,
        fill_mode=FLAGS.fill_mode)


def train(dataset):
    flag_values_dict = FLAGS.flag_values_dict()
    pprint(flag_values_dict, indent=4)
    with open(os.path.join(FLAGS.model, FLAGS_FILENAME), 'w') as f:
        json.dump(flag_values_dict, f, indent=4)

    # FLAGS.weight_ad is parsed to [coverage_min, coverage_max], threshold to apply adaptive weight
    if FLAGS.weight_ad is not None:
        weight_adaptive = [float(x) for x in FLAGS.weight_ad]
    else:
        weight_adaptive = None

    with tf.device('/cpu:0'):
        iter_train, iter_valid = dataset.gen_train_valid(
            n_splits=N_SPLITS, idx_kfold=FLAGS.cv, batch_size=FLAGS.batch_size, adjust=FLAGS.adjust,
            weight_fg=FLAGS.weight_fg, weight_bg=FLAGS.weight_bg, weight_adaptive=weight_adaptive,
            filter_vert_hori=FLAGS.filter_vert_hori, ignore_tiny=FLAGS.ignore_tiny,
            augment_dict=augment_dict())

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,  gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.9, allow_growth=True)))
    K.set_session(sess)

    if FLAGS.debug:
        debug_img_show(iter_train, iter_valid, sess)

    with tf.device('/gpu:0'):
        if FLAGS.restore is not None:
            path_restore = os.path.join(FLAGS.restore, NAME_MODEL)
            model = load_model(path_restore, compile=False)
        elif FLAGS.pretrained is not None:
            model = build_model_pretrained(
                IM_HEIGHT, IM_WIDTH, IM_CHAN, encoder=FLAGS.pretrained,
                spatial_dropout=FLAGS.spatial_dropout)
        elif not FLAGS.use_ref:
            model = build_model(
                IM_HEIGHT, IM_WIDTH, IM_CHAN, batch_norm=FLAGS.batch_norm, drop_out=FLAGS.drop_out)
        else:
            model = build_model_ref(
                IM_HEIGHT, IM_WIDTH, IM_CHAN, batch_norm=FLAGS.batch_norm, drop_out=FLAGS.drop_out,
                depth=FLAGS.depth, start_ch=FLAGS.start_ch)

        model = compile_model(model, optimizer=FLAGS.opt, loss=FLAGS.loss,
                              weight_decay=FLAGS.weight_decay, exclude_bn=FLAGS.exclude_bn)
        write_summary(model, os.path.join(FLAGS.model, MODEL_SUMMARY_FILENAME))
        model.summary()

    path_model = os.path.join(FLAGS.model, NAME_MODEL)

    checkpointer = ModelCheckpoint(path_model, monitor='val_weighted_mean_score', verbose=1, save_best_only=True, mode='max')
    tensorboarder = MyTensorBoard(FLAGS.log, model=model)
    lrscheduler = LearningRateScheduler(
        StepDecay(FLAGS.lr, FLAGS.lr_decay, FLAGS.epochs_decay, FLAGS.freeze_once), verbose=1)

    callbacks = [checkpointer, tensorboarder, lrscheduler]
    if FLAGS.early_stopping:
        callbacks.append(EarlyStopping(patience=5, verbose=1))
    if FLAGS.reduce_on_plateau:
        lrreducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1, mode='min',
                                      epsilon=0.0001, cooldown=4)
        callbacks.append(lrreducer)

    num_train, num_valid = dataset.len_train_valid(n_splits=N_SPLITS, idx_kfold=FLAGS.cv)

    steps_per_epoch = int(num_train / FLAGS.batch_size)
    validation_steps = int(num_valid / FLAGS.batch_size)

    results = model.fit(
        x=iter_train, validation_data=iter_valid,
        epochs=FLAGS.epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
        shuffle=True, callbacks=callbacks)


def debug_img_show(iter_train, iter_valid, sess):
    import numpy as np
    import matplotlib.pyplot as plt

    def show_img_label_mask(images, labels_and_masks, prefix=""):
        num_img = images.shape[0]
        for i, (image, label_and_mask) in enumerate(zip(images, labels_and_masks)):
            print("image.shape is {}".format(image.shape))
            print("label_and_mask.shape is {}".format(label_and_mask.shape))
            image = np.squeeze(image)
            label = label_and_mask[:,:,0]
            mask = label_and_mask[:,:,1]
            plt.title(prefix + "image {}/{}".format(i, num_img))
            plt.imshow(image, cmap='gray', vmin=0, vmax=1)
            plt.colorbar()
            plt.show()

            plt.title(prefix + "label {}/{}".format(i, num_img))
            plt.imshow(label, cmap='gray', vmin=0, vmax=1)
            plt.colorbar()
            plt.show()

            plt.title(prefix + "mask {}/{}".format(i, num_img))
            plt.imshow(mask, cmap='gray', vmin=0, vmax=np.max(mask))
            plt.colorbar()
            plt.show()

    images, labels_and_masks = sess.run(iter_train.get_next())
    show_img_label_mask(images, labels_and_masks, prefix="training ")

    images, labels_and_masks = sess.run(iter_valid.get_next())
    show_img_label_mask(images, labels_and_masks, prefix="validing ")

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