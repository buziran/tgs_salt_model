#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
from absl import app, flags
from sklearn.metrics import confusion_matrix
from PIL import Image
from tqdm import tqdm
import pandas as pd

from constant import *
from dataset import Dataset
from metrics import mean_score_per_image

flags.DEFINE_string(
    'input', '../input/train',
    """path to train mask data""")

flags.DEFINE_string(
    'prediction', '../output/prediction',
    """path to prediction directory""")

flags.DEFINE_integer(
    'cv', 0, help="""index of k-fold cross validation. index must be in 0~9""")

flags.DEFINE_string(
    'score', '../output/score',
    """path to score directory""")

flags.DEFINE_float(
    'threshold', 0.5, """threshold of confidence to predict foreground""")

FLAGS = flags.FLAGS

N_SPLITS = 5


def list_image(path_input):
    sample_ids = next(os.walk(os.path.join(path_input, "masks")))[2]
    sample_ids = sorted(sample_ids)
    return sample_ids


def load_npz(path_pred):
    npzfile = np.load(path_pred)
    return npzfile['arr_0']


def main(argv):
    dataset = Dataset(FLAGS.input)
    train_ids, valid_ids = dataset.kfold_split(N_SPLITS, FLAGS.cv)

    if not os.path.isdir(FLAGS.score):
        os.makedirs(FLAGS.score)

    df = pd.DataFrame(columns=["name", "score", "coverage_true", "coverage_pred"])
    df.astype({"name": str, "score": float, "coverage_true": float, "coverage_pred": float})

    rows = []

    for valid_id in tqdm(valid_ids):
        y_true_path = os.path.join(FLAGS.input, "masks", valid_id)
        y_true = np.array(Image.open(y_true_path)).astype(float)
        y_true = np.round(y_true / 65535.).astype(int)
        y_pred_path = os.path.join(FLAGS.prediction, os.path.splitext(valid_id)[0] + ".npz")
        y_pred = load_npz(y_pred_path)
        score = mean_score_per_image(y_true, y_pred, threshold=FLAGS.threshold)
        coverage_true = np.sum(y_true) / float(ORIG_WIDTH * ORIG_HEIGHT)
        coverage_pred = np.sum(y_pred) / float(ORIG_WIDTH * ORIG_HEIGHT)
        row = {"name": valid_id, "score": score, "coverage_true": coverage_true, "coverage_pred": coverage_pred}
        rows.append(row)

    # Score per image
    path_out = os.path.join(FLAGS.score, "score.csv")
    df = df.append(rows, ignore_index=True)
    df.to_csv(path_out)

    # Score grouped by coverage rate of ground truth
    # bins = np.arange(0, 1.0, 0.05)
    # ind = np.digitize(df['coverage_true'], bins)
    # grouped = df.groupby(ind)

    df = df.sort_values('coverage_true')
    bins = np.arange(0, 1.0, 0.05)
    bins = np.concatenate(([-1e-4], bins))

    grouped = df.groupby(pd.cut(df['coverage_true'], bins))

    path_out = os.path.join(FLAGS.score, "mean.csv")
    mean_df = grouped.mean()
    mean_df.to_csv(path_out)

    path_out = os.path.join(FLAGS.score, "var.csv")
    mean_df = grouped.var()
    mean_df.to_csv(path_out)

if __name__ == '__main__':
    app.run(main)