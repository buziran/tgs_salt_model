#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize
from tqdm import tnrange, tqdm_notebook

from util import RLenc
from input import input_test
from metrics import mean_iou, mean_score, bce_dice_loss
from config import *

tf.flags.DEFINE_string(
    'input', '../input/test',
    """path to test data""")

tf.flags.DEFINE_string(
    'model', 'output/model',
    """path to model directory""")

tf.flags.DEFINE_string(
    'submission', 'output/submission.csv',
    """path to submission file""")

tf.flags.DEFINE_bool(
    'debug', False,
    """Inference only 10 images""")

FLAGS = tf.flags.FLAGS


def main(argv=None):

    if not tf.gfile.Exists(os.path.dirname(FLAGS.submission)):
        tf.gfile.MakeDirs(os.path.dirname(FLAGS.submission))

    test_ids = next(os.walk(os.path.join(FLAGS.input, "images")))[2]
    X_test = input_test(FLAGS.input)

    if FLAGS.debug:
        X_test = X_test[:10]

    path_model = os.path.join(FLAGS.model, NAME_MODEL)
    model = load_model(path_model, custom_objects={'mean_iou': mean_iou, 'mean_score': mean_score}, compile=False)
    model.compile(optimizer="adam", loss='binary_crossentropy', metrics=[mean_iou, mean_score])
    preds_test = model.predict(X_test, verbose=1)

    preds_test_upsampled = []
    for i in tnrange(len(preds_test)):
        preds_test_upsampled.append(
            resize(np.squeeze(preds_test[i]),
                   (ORIG_HEIGHT, ORIG_WIDTH),
                   mode='constant', preserve_range=True))

    # 四捨五入している
    pred_dict = {fn[:-4]: RLenc(np.round(preds_test_upsampled[i])) for i, fn in tqdm_notebook(enumerate(test_ids))}

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(FLAGS.submission)


if __name__ == '__main__':
    tf.app.run()