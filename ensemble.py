#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import subprocess

import os
import tempfile

from scipy.misc import imsave
from tqdm import tqdm
import numpy as np
import pandas as pd

from absl import app, flags

from util import RLenc

flags.DEFINE_string('input', '../input/test', """path to test data""")
flags.DEFINE_string('submission', '../output/submission', """prefix of submission file""")
flags.DEFINE_string('model', '../output/model', """path to model root directory""")
flags.DEFINE_bool('delete', True, """whether to delete temporary directory""")


FLAGS = flags.FLAGS


def list_model(model_root):
    model_dirs = glob.glob(os.path.join(model_root, "**", 'model'), recursive=True)
    model_dirs = filter(lambda x: os.path.isdir(x), model_dirs)
    return model_dirs


class TemporaryDirectory(tempfile.TemporaryDirectory):
    def __init__(self, suffix=None, prefix=None, dir=None, delete=True):
        self.delete = delete
        if self.delete:
            super(TemporaryDirectory, self).__init__(suffix, prefix, dir)
        else:
            self.name = tempfile.mkdtemp(suffix, prefix, dir)

    def __exit__(self, exc, value, tb):
        if self.delete:
            super(TemporaryDirectory, self).__exit__(exc, value, tb)
        else:
            pass


def load_npz(path_pred):
    npzfile = np.load(path_pred)
    return npzfile['arr_0']


def main(argv):

    model_dirs = list_model(FLAGS.model)
    pred_arg_template = ["python", "predict.py", "--input", FLAGS.input, '--npz'] + argv[1:]

    # Predict with each model
    with TemporaryDirectory(prefix="pred-", delete=FLAGS.delete) as tdir:
        print("Temporary directory {} is created".format(tdir))
        path_preds = []
        for d in model_dirs:
            dirname = os.path.basename(os.path.dirname(d))
            path_pred = os.path.join(tdir, dirname)
            pred_arg = pred_arg_template + ["--model", d, "--prediction", path_pred]
            subprocess.run(pred_arg)
            path_preds.append(path_pred)

        fn_dict = {"min": np.min, "max": np.max, "mean": np.mean, "median": np.median}
        for suffix, fn in fn_dict.items():
            output_file = FLAGS.submission + "_" + suffix + ".csv"
            if FLAGS.delete:
                img_dir = None
            else:
                img_dir = os.path.join(tdir, 'ensemble-{}'.format(suffix))
                os.makedirs(img_dir, exist_ok=True)

            ensemble_pred(path_preds, output_file, fn, img_dir)


def ensemble_pred(path_preds, output_file, fn, img_dir=None):
    pred_dict = {}
    pred_files = list(filter(lambda x: x.endswith('.npz'), os.listdir(path_preds[0])))

    for pred_file in tqdm(pred_files):
        preds = []
        for d in path_preds:
            path_pred = os.path.join(d, pred_file)
            pred = load_npz(path_pred)
            preds.append(pred)
        preds = np.stack(preds, axis=2).astype(np.float)
        ensembled = fn(preds, axis=2)

        pred_dict.update({pred_file[:-4]: RLenc(np.round(ensembled))})

        if img_dir is not None:
            y_pred = np.clip(ensembled * 255, 0, 255).astype(np.uint8)
            filename = os.path.join(img_dir, os.path.splitext(pred_file)[0] + '.png')
            imsave(filename, y_pred)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sub = pd.DataFrame.from_dict(pred_dict, orient='index')
    sub.index.names = ['id']
    sub.columns = ['rle_mask']
    sub.to_csv(output_file)


if __name__ == '__main__':
    app.run(main)
