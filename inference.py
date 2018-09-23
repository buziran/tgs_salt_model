#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pandas as pd
import numpy as np
from skimage.util import crop
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf
from skimage.transform import resize
from tqdm import tnrange, tqdm_notebook, tqdm

from util import RLenc
from dataset import Dataset
from metrics import mean_iou, mean_score, weighted_bce_dice_loss
from constant import *

tf.flags.DEFINE_string(
    'input', '../input/test',
    """path to test data""")

tf.flags.DEFINE_string(
    'model', '../output/model',
    """path to model directory""")

tf.flags.DEFINE_string(
    'submission', '../output/submission.csv',
    """path to submission file""")

tf.flags.DEFINE_integer(
    'batch_size', 32, """batch size""")

tf.flags.DEFINE_bool(
    'debug', False,
    """Inference only 10 images""")

tf.flags.DEFINE_enum(
    'adjust', 'resize', enum_values=['resize', 'resize-cv', 'pad'],
    help="""mode to adjust image size""")

tf.flags.DEFINE_float(
    'threshold', 0.5, """threshold of confidence to predict foreground""")

FLAGS = tf.flags.FLAGS


def main(argv=None):

    if not tf.gfile.Exists(os.path.dirname(FLAGS.submission)):
        tf.gfile.MakeDirs(os.path.dirname(FLAGS.submission))

    dataset = Dataset(FLAGS.input)
    iter_test  = dataset.gen_test(batch_size=FLAGS.batch_size, adjust=FLAGS.adjust)

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,  gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.9, allow_growth=True)))
    K.set_session(sess)

    path_model = os.path.join(FLAGS.model, NAME_MODEL)
    model = load_model(path_model, compile=False)
    model.compile(optimizer="adam", loss='binary_crossentropy')

    num_batch = int(np.ceil(len(dataset) / FLAGS.batch_size))
    sample_tensor = iter_test.get_next()

    preds_test_upsampled = []
    test_ids = []
    for id_batch in tqdm(range(num_batch)):
        xs, paths = sess.run(sample_tensor)
        ids = np.asarray([os.path.split(path)[1].decode() for path in paths])
        test_ids.extend(ids)
        ys_pred = model.predict_on_batch(xs)

        for pred in ys_pred:
            pred = np.squeeze(pred)
            if FLAGS.adjust in ['resize']:
                pred = resize(pred, (ORIG_HEIGHT, ORIG_WIDTH), mode='constant', preserve_range=True)
            elif FLAGS.adjust in ['reflect', 'constant', 'symmetric']:
                height_padding = ((IM_HEIGHT - ORIG_HEIGHT) // 2, IM_HEIGHT - ORIG_HEIGHT - (IM_HEIGHT - ORIG_HEIGHT) // 2)
                width_padding = ((IM_WIDTH - ORIG_WIDTH) // 2, IM_WIDTH - ORIG_WIDTH - (IM_WIDTH - ORIG_WIDTH) // 2)
                pred = crop(pred, (height_padding, width_padding))
            preds_test_upsampled.append(pred)

    # 四捨五入している
    pred_dict = {fn[:-4]: RLenc((preds_test_upsampled[i] > FLAGS.threshold).astype(np.float)) for i, fn in tqdm_notebook(enumerate(test_ids))}

    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(FLAGS.submission)


if __name__ == '__main__':
    tf.app.run()