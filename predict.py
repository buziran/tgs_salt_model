#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tqdm import tqdm
from scipy.misc import imsave
from skimage.transform import resize
from skimage.util import crop
import pandas as pd

from dataset import Dataset
from metrics import mean_iou, mean_score
from constant import *
from util import get_metrics, get_custom_objects, sigmoid

tf.flags.DEFINE_string(
    'input', '../input/train',
    """path to test data""")

tf.flags.DEFINE_string(
    'model', '../output/model',
    """path to model directory""")

tf.flags.DEFINE_integer(
    'batch_size', 32, """batch size""")

tf.flags.DEFINE_string(
    'prediction', '../output/prediction',
    """path to prediction directory""")

tf.flags.DEFINE_bool(
    'npz', True,
    """whether to save as npz""")

tf.flags.DEFINE_enum(
    'adjust', 'symmetric', enum_values=['resize', 'reflect', 'constant', 'symmetric'], help="""mode to adjust image size""")

tf.flags.DEFINE_bool('deep_supervised', False, """whether to use deep-supervised model""")

tf.flags.DEFINE_bool('with_depth', False, """whether to use depth information""")

FLAGS = tf.flags.FLAGS

FILENAME_IMAGE_PREDS = "image_preds.csv"


def save_png(ys_pred, ids, path_out, adjust='resize'):
    """Save confidence image as uint.8"""
    ys_pred = np.clip(ys_pred * 255, 0, 255)
    ys_pred = np.squeeze(ys_pred.astype(np.uint8), axis=3)
    ids = ids.astype(str)
    for y_pred, id in zip(ys_pred, ids):
        if adjust in ['resize']:
            y_pred = resize(y_pred, (ORIG_HEIGHT, ORIG_WIDTH))
        elif adjust in ['reflect', 'constant', 'symmetric']:
            height_padding = ((IM_HEIGHT - ORIG_HEIGHT) // 2, IM_HEIGHT - ORIG_HEIGHT - (IM_HEIGHT - ORIG_HEIGHT) // 2)
            width_padding = ((IM_WIDTH - ORIG_WIDTH) // 2, IM_WIDTH - ORIG_WIDTH - (IM_WIDTH - ORIG_WIDTH) // 2)
            y_pred = crop(y_pred, (height_padding, width_padding))
        filename = os.path.join(path_out, id)
        imsave(filename, y_pred)


def save_npz(ys_pred, ids, path_out, adjust='resize'):
    ids = [os.path.splitext(id)[0] + '.npz' for id in ids]
    ys_pred = np.squeeze(ys_pred, axis=3)
    for y_pred, id in zip(ys_pred, ids):
        if adjust in ['resize']:
            y_pred = resize(y_pred, (ORIG_HEIGHT, ORIG_WIDTH))
        elif adjust in ['reflect', 'constant', 'symmetric']:
            height_padding = ((IM_HEIGHT - ORIG_HEIGHT) // 2, IM_HEIGHT - ORIG_HEIGHT - (IM_HEIGHT - ORIG_HEIGHT) // 2)
            width_padding = ((IM_WIDTH - ORIG_WIDTH) // 2, IM_WIDTH - ORIG_WIDTH - (IM_WIDTH - ORIG_WIDTH) // 2)
            y_pred = crop(y_pred, (height_padding, width_padding))
        filename = os.path.join(path_out, id)
        np.savez(filename, y_pred)


def main(argv=None):

    tf.gfile.MakeDirs(FLAGS.prediction)

    dataset = Dataset(FLAGS.input)
    iter_test  = dataset.gen_test(batch_size=FLAGS.batch_size, adjust=FLAGS.adjust, with_depth=FLAGS.with_depth)

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,  gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.9, allow_growth=True)))
    K.set_session(sess)

    path_model = os.path.join(FLAGS.model, NAME_MODEL)
    model = load_model(path_model, compile=False)
    model.compile(optimizer="adam", loss='binary_crossentropy')

    num_batch = int(np.ceil(len(dataset) / FLAGS.batch_size))
    sample_tensor = iter_test.get_next()
    image_preds = {}
    for id_batch in tqdm(range(num_batch)):
        xs, paths = sess.run(sample_tensor)
        ids = np.asarray([os.path.split(path)[1].decode() for path in paths])

        if id_batch == num_batch:
            break
        ys_outputs = model.predict_on_batch(xs)
        if not FLAGS.deep_supervised:
            ys_logits = ys_outputs
            ys_pred = sigmoid(ys_logits)
            save_png(ys_pred, ids, FLAGS.prediction, FLAGS.adjust)
            if FLAGS.npz:
                save_npz(ys_pred, ids, FLAGS.prediction, FLAGS.adjust)
        else:
            ys_logits, image_logits = ys_outputs[0], ys_outputs[2]
            image_pred = sigmoid(image_logits)
            image_preds.update({i: p for i, p in zip(ids, image_pred)})
            ys_pred = sigmoid(ys_logits)
            save_png(ys_pred, ids, FLAGS.prediction, FLAGS.adjust)
            if FLAGS.npz:
                save_npz(ys_pred, ids, FLAGS.prediction, FLAGS.adjust)

    if FLAGS.deep_supervised:
        df_image_preds = pd.DataFrame.from_dict(image_preds, orient='index')
        df_image_preds.to_csv(os.path.join(FLAGS.prediction, FILENAME_IMAGE_PREDS))


if __name__ == '__main__':
    tf.app.run()