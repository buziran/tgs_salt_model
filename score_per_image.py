#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
from absl import app, flags
from sklearn.metrics import confusion_matrix
from PIL import Image
from sklearn.model_selection import KFold
from tqdm import tqdm
import pandas as pd

from constant import *

flags.DEFINE_string(
    'input', '../input/train',
    """path to train mask data""")

flags.DEFINE_string(
    'prediction', '../output/prediction/train',
    """path to prediction directory""")

flags.DEFINE_integer(
    'cv', 0, help="""index of k-fold cross validation. index must be in 0~9""")

flags.DEFINE_string(
    'score', '../output/score',
    """path to score directory""")

FLAGS = flags.FLAGS

N_SPLITS = 10


def mean_score(y_true, y_pred):
    """Calculate score per image"""
    # GT, Predともに前景ゼロの場合はスコアを1とする
    if np.any(y_true) == False and np.any(y_pred) == False:
        return 1.

    threasholds_iou = np.arange(0.5, 1.0, 0.05, dtype=float)
    y_true = np.reshape(y_true, (-1))
    y_pred = np.reshape(y_pred, (-1))
    total_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    sum_over_row = np.sum(total_cm, 0).astype(float)
    sum_over_col = np.sum(total_cm, 1).astype(float)
    cm_diag = np.diag(total_cm).astype(float)
    denominator = sum_over_row + sum_over_col - cm_diag
    denominator = np.where(np.greater(denominator, 0), denominator, np.ones_like(denominator))
    # iou[0]: 背景のIoU
    # iou[1]: 前景のIoU
    iou = np.divide(cm_diag, denominator)
    iou_fg = iou[1]
    greater = np.greater(iou_fg, threasholds_iou)
    score_per_image = np.mean(greater.astype(float))
    return score_per_image


def list_image(path_input):
    sample_ids = next(os.walk(os.path.join(path_input, "masks")))[2]
    sample_ids = sorted(sample_ids)
    return sample_ids


def load_npz(path_pred):
    npzfile = np.load(path_pred)
    return npzfile['arr_0']


def split_train_valid(sample_ids, n_splits, idx_kfold):
    kf = KFold(n_splits)
    num_samples = len(sample_ids)
    for idx, (train_index, valid_index) in enumerate(kf.split(range(num_samples))):
        if idx == idx_kfold:
            break
    train_ids = np.array(sample_ids)[train_index]
    valid_ids = np.array(sample_ids)[train_index]
    return train_ids, valid_ids


def main(argv):
    sample_ids = list_image(FLAGS.input)
    train_ids, valid_ids = split_train_valid(sample_ids, N_SPLITS, FLAGS.cv)

    if not os.path.isdir(FLAGS.score):
        os.makedirs(FLAGS.score)

    df = pd.DataFrame(columns=["name", "score", "coverage_true", "coverage_pred"])
    df.astype({"name": str, "score": float, "coverage_true": float, "coverage_pred": float})

    rows = []

    for train_id in tqdm(train_ids):
        y_true_path = os.path.join(FLAGS.input, "masks", train_id)
        y_true = np.array(Image.open(y_true_path)).astype(float)
        y_true = np.round(y_true / 65535.).astype(int)
        y_pred_path = os.path.join(FLAGS.prediction, os.path.splitext(train_id)[0] + ".npz")
        y_pred = load_npz(y_pred_path)
        y_pred = np.round(y_pred).astype(int)
        score = mean_score(y_true, y_pred)
        coverage_true = np.sum(y_true) / float(ORIG_WIDTH * ORIG_HEIGHT)
        coverage_pred = np.sum(y_pred) / float(ORIG_WIDTH * ORIG_HEIGHT)
        row = {"name": train_id, "score": score, "coverage_true": coverage_true, "coverage_pred": coverage_pred}
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
    print(bins)

    grouped = df.groupby(pd.cut(df['coverage_true'], bins))

    path_out = os.path.join(FLAGS.score, "mean.csv")
    mean_df = grouped.mean()
    mean_df.to_csv(path_out)

    path_out = os.path.join(FLAGS.score, "var.csv")
    mean_df = grouped.var()
    mean_df.to_csv(path_out)

if __name__ == '__main__':
    app.run(main)