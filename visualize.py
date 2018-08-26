#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tqdm import tnrange, tqdm_notebook, tqdm
import matplotlib.pyplot as plt

from input import Dataset
from metrics import mean_iou, mean_score
from constant import *

tf.flags.DEFINE_string(
    'input', '../input/train',
    """path to test data""")

tf.flags.DEFINE_string(
    'model', '../output/model',
    """path to model directory""")

tf.flags.DEFINE_bool(
    'cv', 0,
    """Inference only 10 images""")

tf.flags.DEFINE_bool('dice', True, """whether to use dice loss""")

tf.flags.DEFINE_integer(
    'batch_size', 16, """batch size""")

tf.flags.DEFINE_string(
    'prediction', 'output/prediction',
    """path to prediction directory""")

FLAGS = tf.flags.FLAGS


def create_image(xs, ys_true, ys_pred, ids, path_out):
    num_images = xs.shape[0]
    grid_width = 4 * 3
    grid_height = int(num_images / 4)

    xs = np.squeeze(xs, axis=3)
    ys_pred = np.squeeze(ys_pred, axis=3)
    ys_true = np.squeeze(ys_true, axis=3)

    fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))
    plt.subplots_adjust(wspace=0.2, hspace=0.6)
    for i, (x, y_true, y_pred, id) in enumerate(zip(xs, ys_true, ys_pred, ids)):
        # raw image
        ax = axs[int(i * 3 / grid_width), (i * 3) % grid_width]
        ax.imshow(x, cmap="Greys")
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        # raw image + ground truth
        ax = axs[int(i * 3 / grid_width), (i * 3) % grid_width + 1]
        ax.set_title(id)
        ax.imshow(x, cmap="Greys")
        ax.imshow(y_true, alpha=0.3, cmap="Greens")
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        # raw image + prediction
        ax = axs[int(i * 3 / grid_width), (i * 3) % grid_width + 2]
        ax.imshow(x, cmap="Greys")
        ax.imshow(y_pred, alpha=0.3, cmap="Blues")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
    plt.suptitle("Green: ground truth. Blue: prediction.")
    fig.savefig(path_out)
    plt.close()


def main(argv=None):

    if tf.gfile.Exists(FLAGS.prediction):
        tf.gfile.DeleteRecursively(FLAGS.prediction)
    tf.gfile.MakeDirs(os.path.join(FLAGS.prediction, "train"))
    tf.gfile.MakeDirs(os.path.join(FLAGS.prediction, "valid"))

    dataset = Dataset(FLAGS.input)

    train_generator, valid_generator = dataset.create_generator_cv(
        n_splits=N_SPLITS, idx_kfold=FLAGS.cv, batch_size=FLAGS.batch_size, augment_dict={}, shuffle=False, with_id=True)

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,  gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.9, allow_growth=True)))
    K.set_session(sess)

    path_model = os.path.join(FLAGS.model, NAME_MODEL)
    model = load_model(path_model, custom_objects={'mean_iou': mean_iou, 'mean_score': mean_score}, compile=False)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=[mean_iou, mean_score])

    for id_batch, ((xs, ids), ys_true) in enumerate(tqdm(valid_generator, total=dataset.num_valid // FLAGS.batch_size)):
        if id_batch == dataset.num_valid // FLAGS.batch_size:
            break
        ys_pred = model.predict_on_batch(xs)
        path_out = os.path.join(FLAGS.prediction, "valid", "{}.png".format(id_batch))
        create_image(xs, ys_true, ys_pred, ids, path_out)
    print("Finish visualize of validation data")

    for id_batch, ((xs, ids), ys_true) in enumerate(tqdm(train_generator, total=dataset.num_train // FLAGS.batch_size)):
        if id_batch == dataset.num_train / FLAGS.batch_size:
            break
        ys_pred = model.predict_on_batch(xs)
        path_out = os.path.join(FLAGS.prediction, "train", "{}.png".format(id_batch))
        create_image(xs, ys_true, ys_pred, ids, path_out)
    print("Finish visualize of training data")


if __name__ == '__main__':
    tf.app.run()