#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow.keras.losses

from input import input_train, input_test, Dataset
from metrics import mean_iou, mean_score, bce_dice_loss
from config import *

tf.flags.DEFINE_string(
    'input', "../input/train",
    """path to train data""")

tf.flags.DEFINE_string(
    'model', './output/model',
    """path to model directory""")

tf.flags.DEFINE_integer(
    'cv', 0,
    """index of k-fold cross validation. index must be in 0~9""")

tf.flags.DEFINE_bool('dice', True, """whether to use dice loss""")

FLAGS = tf.flags.FLAGS

tensorflow.keras.losses.bce_dice_loss = bce_dice_loss


def eval(dataset):

    path_model = os.path.join(FLAGS.model, name_model)

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