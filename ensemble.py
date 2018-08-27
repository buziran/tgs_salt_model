#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import subprocess

import os
import tempfile

from PIL import Image
from scipy.misc import imsave
from tqdm import tqdm
import numpy as np
import pandas as pd

from absl import app, flags

from util import RLenc

flags.DEFINE_string('input', '../input/test', """path to test data""")
flags.DEFINE_string('submission', '../output/submission.csv', """path to submission file""")
flags.DEFINE_string('model', '../output/model', """path to model root directory""")
flags.DEFINE_bool('delete', True, """whether to delete temporary directory""")
flags.DEFINE_enum('type', 'mean', enum_values=['mean', 'max', 'min', 'median'], help="""ensemble type""")


FLAGS = flags.FLAGS


def list_model(model_root):
    model_dirs = glob.glob(os.path.join(model_root, "**", 'model'))
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


def main(argv):

    model_dirs = list_model(FLAGS.model)
    pred_arg_template = ["python", "predict.py", "--input", FLAGS.input]

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

        pred_dict = {}
        image_names = os.listdir(path_preds[0])

        path_ensembled = os.path.join(tdir, "ensemble-preds")
        os.makedirs(path_ensembled)
        for image_name in tqdm(image_names):
            images = []
            for d in path_preds:
                im = np.asarray(Image.open(os.path.join(d, image_name)))
                images.append(im)
            images = np.stack(images, axis=2).astype(np.float) / 255.
            if FLAGS.type == 'mean':
                ensembled = np.mean(images, axis=2)
            elif FLAGS.type == 'max':
                ensembled = np.max(images, axis=2)
            elif FLAGS.type == 'min':
                ensembled = np.min(images, axis=2)
            elif FLAGS.type == 'median':
                ensembled = np.median(images, axis=2)

            pred_dict.update({image_name[:-4]: RLenc(np.round(ensembled))})

            if not FLAGS.delete:
                y_pred = np.clip(ensembled * 255, 0, 255)
                filename = os.path.join(path_ensembled, image_name)
                imsave(filename, y_pred)


        sub = pd.DataFrame.from_dict(pred_dict, orient='index')
        sub.index.names = ['id']
        sub.columns = ['rle_mask']
        sub.to_csv(FLAGS.submission)


if __name__ == '__main__':
    app.run(main)
