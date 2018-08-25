#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow.keras.losses

from input import Dataset
from metrics import mean_iou, mean_score, bce_dice_loss
from constant import *
import eval_config

FLAGS = tf.flags.FLAGS

tensorflow.keras.losses.bce_dice_loss = bce_dice_loss


def eval(dataset):

    path_model = os.path.join(FLAGS.model, NAME_MODEL)

    config = tf.ConfigProto(
        allow_soft_placement=True,  gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.9, allow_growth=True))

    train_generator, valid_generator = dataset.create_generator(
        n_splits=N_SPLITS, idx_kfold=FLAGS.cv, batch_size=BATCH_SIZE, augment_dict={}, shuffle=False)

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            K.set_session(sess)
            model = load_model(path_model, custom_objects={'mean_iou': mean_iou, 'mean_score': mean_score}, compile=False)
            if FLAGS.dice:
                loss = bce_dice_loss
            else:
                loss = 'binary_crossentropy'
            model.compile(optimizer="adam", loss=loss, metrics=[mean_iou, mean_score])

            steps_train = dataset.num_train / BATCH_SIZE
            steps_valid = dataset.num_valid / BATCH_SIZE
            metrics = model.evaluate_generator(train_generator, steps=steps_train, max_queue_size=dataset.num_train)
            print("Training loss:{}, iou:{}, score:{}".format(metrics[0], metrics[1], metrics[2]))
            metrics = model.evaluate_generator(valid_generator, steps=steps_valid, max_queue_size=dataset.num_train)
            print("Validation loss:{}, iou:{}, score:{}".format(metrics[0], metrics[1], metrics[2]))


def main(argv=None):
    dataset = Dataset(FLAGS.input)
    eval(dataset)


if __name__ == '__main__':
    tf.app.run()