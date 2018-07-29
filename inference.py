#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage.transform import resize
from tqdm import tnrange, tqdm_notebook

from util import RLenc
from input import input_test
from metrics import mean_iou
from config import *

tf.flags.DEFINE_string(
    'input', None,
    """path to test data""")

tf.flags.DEFINE_string(
    'model', 'output/model',
    """path to model directory""")

tf.flags.DEFINE_string(
    'submission', 'output/submission.csv',
    """path to submission file""")

FLAGS = tf.flags.FLAGS


def main(argv=None):

    if not tf.gfile.Exists(os.path.dirname(FLAGS.submission)):
        tf.gfile.MakeDirs(os.path.dirname(FLAGS.submission))

    test_ids = next(os.walk(os.path.join(FLAGS.input, "images")))[2]
    X_test = input_test(FLAGS.input)
    path_model = os.path.join(FLAGS.model, name_model)
    model = load_model(path_model, custom_objects={'mean_iou': mean_iou})
    preds_test = model.predict(X_test, verbose=1)

    preds_test_upsampled = []
    for i in tnrange(len(preds_test)):
        preds_test_upsampled.append(
            resize(np.squeeze(preds_test[i]),
                   (orig_height, orig_width),
                   mode='constant', preserve_range=True))

    pred_dict = {fn[:-4]: RLenc(np.round(preds_test_upsampled[i])) for i, fn in tqdm_notebook(enumerate(test_ids))}

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(FLAGS.submission)


if __name__ == '__main__':
    tf.app.run()