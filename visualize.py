#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.python.framework.errors_impl import OutOfRangeError
from tqdm import tnrange, tqdm_notebook, tqdm
import matplotlib.pyplot as plt
from util import load_npz
from dataset import Dataset
from metrics import mean_iou, mean_score, mean_score_per_image
from constant import *

tf.flags.DEFINE_string(
    'input', '../input/train',
    """path to test data""")

tf.flags.DEFINE_string(
    'prediction', '../output/prediction',
    """path to prediction directory""")

tf.flags.DEFINE_bool('dice', True, """whether to use dice loss""")

tf.flags.DEFINE_integer(
    'batch_size', 16, """batch size""")

tf.flags.DEFINE_string(
    'visualize', '../output/visualize',
    """path to prediction directory""")

tf.flags.DEFINE_integer(
    'cv', 0, help="""index of k-fold cross validation. index must be in 0~9""")

FLAGS = tf.flags.FLAGS


def create_image(x, y_true, y_pred, id, path_out, title=None):
    x = np.reshape(x, (ORIG_WIDTH, ORIG_HEIGHT, IM_CHAN)).astype(np.uint8)
    y_true = np.reshape(y_true, (ORIG_WIDTH, ORIG_HEIGHT)).astype(np.uint8)
    y_pred = np.reshape(y_pred, (ORIG_WIDTH, ORIG_HEIGHT)).astype(np.uint8)

    fig, axs = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.2, hspace=0.6)

    # raw image
    ax = axs[0]
    ax.imshow(x, cmap="Greys")
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # raw image + ground truth
    ax = axs[1]
    ax.imshow(x, cmap="Greys")
    ax.imshow(y_true, alpha=0.3, cmap="Greens")
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # raw image + prediction
    ax = axs[2]
    ax.imshow(x, cmap="Greys")
    ax.imshow(y_pred, alpha=0.3, cmap="Blues")
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    if title == None:
        title = "Green: ground truth. Blue: prediction."
    plt.suptitle(title)
    fig.savefig(path_out)
    plt.close()


def load_preds(dir_pred, id_samples):
    preds = []
    for id in id_samples:
        f = os.path.join(dir_pred, id)
        pred = load_npz(f)
        preds.append(pred)

    return preds


def main(argv=None):
    if tf.gfile.Exists(FLAGS.visualize):
        tf.gfile.DeleteRecursively(FLAGS.visualize)
    tf.gfile.MakeDirs(FLAGS.visualize)

    dataset = Dataset(FLAGS.input)
    iter_valid = dataset.gen_valid(n_splits=N_SPLITS, idx_kfold=FLAGS.cv, adjust='never', with_path=True, repeat=1)

    X_valids = []
    Y_valids = []
    path_valids = []
    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        try:
            while True:
                xs, ys, ids = sess.run(iter_valid.get_next())
                X_valids.append(xs)
                Y_valids.append(ys)
                path_valids.extend(ids)
        except OutOfRangeError:
            pass
    X_valids = np.concatenate(X_valids)
    Y_valids = np.concatenate(Y_valids)
    id_valids = [os.path.basename(id.decode('utf-8')) for id in path_valids]

    pred_filenames = [os.path.basename(id.replace('png', 'npz')) for id in id_valids]
    pred_valids = load_preds(FLAGS.prediction, pred_filenames)

    for id, x, y_true, y_pred in tqdm(zip(id_valids, X_valids, Y_valids, pred_valids), total=len(id_valids)):
        path_out = os.path.join(FLAGS.visualize, id)

        # Make tile
        score = mean_score_per_image(y_true, y_pred)
        title = "id:{}, score:{}".format(id, score)

        create_image(
            np.clip(x * 255, 0, 255), np.clip(y_true * 255, 0, 255), np.clip(y_pred * 255, 0, 255), id, path_out, title=title)
    print("Finish visualize of validation data")



if __name__ == '__main__':
    tf.app.run()