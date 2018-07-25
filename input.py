import os
import sys

import numpy as np
from tqdm import tqdm_notebook
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from config import *


def input_train(path_train):
    train_ids = next(os.walk(os.path.join(path_train, "images")))[2]

    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm_notebook(enumerate(train_ids), total=len(train_ids)):
        path = path_train
        img = load_img(path + '/images/' + id_)
        x = img_to_array(img)[:, :, 1]
        x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
        X_train[n] = x
        mask = img_to_array(load_img(path + '/masks/' + id_))[:, :, 1]
        Y_train[n] = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

    print('Done!')

    return X_train, Y_train


def input_test(path_test):
    test_ids = next(os.walk(os.path.join(path_test, "images")))[2]

    X_test = np.zeros((len(test_ids), im_height, im_width, im_chan), dtype=np.uint8)
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm_notebook(enumerate(test_ids), total=len(test_ids)):
        path = path_test
        img = load_img(path + '/images/' + id_)
        x = img_to_array(img)[:, :, 1]
        x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
        X_test[n] = x

    print('Done!')

    return X_test


if __name__ == '__main__':
    import numpy.random as random
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='path_train', help="root of train images/masks")
    args = parser.parse_args()

    X_train, Y_train = input(args.path_train)

    train_ids = next(os.walk(args.path_train + "images"))[2]
    ix = random.randint(0, len(train_ids))
    plt.imshow(np.dstack((X_train[ix], X_train[ix], X_train[ix])))
    plt.show()
    tmp = np.squeeze(Y_train[ix]).astype(np.float32)
    plt.imshow(np.dstack((tmp, tmp, tmp)))
    plt.show()

