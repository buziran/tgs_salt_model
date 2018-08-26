#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tqdm import tqdm
from scipy.misc import imsave

from input import Dataset
from metrics import mean_iou, mean_score
from constant import *

tf.flags.DEFINE_string(
    'input', '../input/train',
    """path to test data""")

tf.flags.DEFINE_string(
    'model', 'output/model',
    """path to model directory""")

tf.flags.DEFINE_integer(
    'batch_size', 32, """batch size""")

tf.flags.DEFINE_string(
    'prediction', 'output/prediction/train',
    """path to prediction directory""")

FLAGS = tf.flags.FLAGS


def save_prediction(ys_pred, ids, path_out):
    """Save confidence image as uint.8"""
    ys_pred = np.clip(ys_pred * 255, 0, 255)
    ys_pred = np.squeeze(ys_pred.astype(np.uint8), axis=3)
    ids = ids.astype(str)
    for y_pred, id in zip(ys_pred, ids):
        filename = os.path.join(path_out, id)
        imsave(filename, y_pred)


def main(argv=None):

    if tf.gfile.Exists(FLAGS.prediction):
        tf.gfile.DeleteRecursively(FLAGS.prediction)
    tf.gfile.MakeDirs(FLAGS.prediction)

    dataset = Dataset(FLAGS.input)

    sample_generator = dataset.create_generator(batch_size=FLAGS.batch_size, augment_dict={}, shuffle=False, with_id=True)

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,  gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.9, allow_growth=True)))
    K.set_session(sess)

    path_model = os.path.join(FLAGS.model, NAME_MODEL)
    model = load_model(path_model, custom_objects={'mean_iou': mean_iou, 'mean_score': mean_score}, compile=False)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=[mean_iou, mean_score])

    for id_batch, ((xs, ids), ys_true) in enumerate(
            tqdm(sample_generator, total=dataset.num_samples // FLAGS.batch_size)):

        if id_batch == dataset.num_samples // FLAGS.batch_size:
            break
        ys_pred = model.predict_on_batch(xs)
        save_prediction(ys_pred, ids, FLAGS.prediction)


if __name__ == '__main__':
    tf.app.run()