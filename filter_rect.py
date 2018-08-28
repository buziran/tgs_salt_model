#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import shutil
import subprocess
import os
import tempfile
import json

from PIL import Image
from tqdm import tqdm
import numpy as np

from absl import app, flags

flags.DEFINE_string('input', '../input/train', """path to test data""")
flags.DEFINE_string('filtered', '../input/filtered', """path to filtered data""")
flags.DEFINE_string('rectangle', '../input/rectangle', """path to rectangle mask data""")

FLAGS = flags.FLAGS

def is_rectangle(im):
    if np.all(im == 255) or np.all(im == 0):
        return False

    col_mean = np.mean(im, axis=0).astype(np.uint8)
    if np.all(np.isin(col_mean, [0, 255])):
        return True
    row_mean = np.mean(im, axis=1).astype(np.uint8)
    if np.all(np.isin(row_mean, [0, 255])):
        return True


def main(argv):
    if os.path.isdir(FLAGS.filtered):
        shutil.rmtree(FLAGS.filtered)
    os.makedirs(os.path.join(FLAGS.filtered, "images"))
    os.makedirs(os.path.join(FLAGS.filtered, "masks"))

    if os.path.isdir(FLAGS.rectangle):
        shutil.rmtree(FLAGS.rectangle)
    os.makedirs(os.path.join(FLAGS.rectangle, "images"))
    os.makedirs(os.path.join(FLAGS.rectangle, "masks"))

    path_masks = glob.glob(os.path.join(FLAGS.input, "masks", "*.png"))
    for path_mask in tqdm(path_masks):
        filename = os.path.basename(path_mask)
        im = Image.open(path_mask).convert('L')
        im = np.asarray(im)

        src_img = os.path.join(FLAGS.input, "images", filename)
        src_msk = os.path.join(FLAGS.input, "masks", filename)

        if not is_rectangle(im):
            dst_img = os.path.join(FLAGS.filtered, "images", filename)
            dst_msk = os.path.join(FLAGS.filtered, "masks", filename)
        else:
            dst_img = os.path.join(FLAGS.rectangle, "images", filename)
            dst_msk = os.path.join(FLAGS.rectangle, "masks", filename)

        os.symlink(os.path.abspath(src_img), dst_img)
        os.symlink(os.path.abspath(src_msk), dst_msk)


if __name__ == '__main__':
    app.run(main)

