import os
import sys

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm_notebook
from skimage.transform import resize
from skimage.util import pad
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from constant import *


class Dataset(object):
    def __init__(self, path_input):
        self.path_input = path_input


    def load_train(self, adjust='resize'):
        train_ids = next(os.walk(os.path.join(self.path_input, "images")))[2]
        train_ids = sorted(train_ids)

        # Get and resize train images and masks
        if adjust in ['resize', 'pad']:
            im_height, im_width = IM_HEIGHT, IM_WIDTH
        elif adjust in ['never']:
            im_height, im_width = ORIG_HEIGHT, ORIG_WIDTH

        X_samples = np.zeros((len(train_ids), im_height, im_width, IM_CHAN), dtype=np.uint8)
        Y_samples = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool)
        print('Getting and resizing train images and masks ... ')
        sys.stdout.flush()
        for n, id_ in tqdm_notebook(enumerate(train_ids), total=len(train_ids)):
            path = self.path_input
            img = load_img(path + '/images/' + id_)
            x = img_to_array(img)[:, :, 1]
            mask = img_to_array(load_img(path + '/masks/' + id_))[:, :, 1]
            if adjust == 'resize':
                X_samples[n] = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
                Y_samples[n] = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)
            elif adjust == 'pad':
                height_padding = ((IM_HEIGHT - ORIG_HEIGHT) // 2, IM_HEIGHT - ORIG_HEIGHT - (IM_HEIGHT - ORIG_HEIGHT) // 2)
                width_padding = ((IM_WIDTH - ORIG_WIDTH) // 2, IM_WIDTH - ORIG_WIDTH - (IM_WIDTH - ORIG_WIDTH) // 2)
                x = pad(x, (height_padding, width_padding), mode='reflect')
                mask = pad(mask, (height_padding, width_padding), mode='reflect')
                X_samples[n] = np.reshape(x, newshape=(im_height, im_width, IM_CHAN))
                Y_samples[n] = np.reshape(mask, newshape=(im_height, im_width, 1))
            elif adjust == 'never':
                X_samples[n] = np.reshape(x, newshape=(im_height, im_width, IM_CHAN))
                Y_samples[n] = np.reshape(mask, newshape=(im_height, im_width, 1))

        print('Done!')

        id_samples = np.array(train_ids)

        self.id_samples = id_samples
        self.X_samples = X_samples
        self.Y_samples = Y_samples

    def load_test(self, adjust='resize'):
        test_ids = next(os.walk(os.path.join(self.path_input, "images")))[2]
        test_ids = sorted(test_ids)

        # Get and resize train images and masks
        if adjust in ['resize', 'pad']:
            im_height, im_width = IM_HEIGHT, IM_WIDTH
        elif adjust in ['never']:
            im_height, im_width = ORIG_HEIGHT, ORIG_WIDTH

        X_samples = np.zeros((len(test_ids), im_height, im_width, IM_CHAN), dtype=np.uint8)
        print('Getting and resizing test images ... ')
        sys.stdout.flush()
        for n, id_ in tqdm_notebook(enumerate(test_ids), total=len(test_ids)):
            path = self.path_input
            img = load_img(path + '/images/' + id_)
            x = img_to_array(img)[:, :, 1]
            if adjust == 'resize':
                X_samples[n] = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
            elif adjust == 'pad':
                height_padding = ((IM_HEIGHT - ORIG_HEIGHT) // 2, IM_HEIGHT - ORIG_HEIGHT - (IM_HEIGHT - ORIG_HEIGHT) // 2)
                width_padding = ((IM_WIDTH - ORIG_WIDTH) // 2, IM_WIDTH - ORIG_WIDTH - (IM_WIDTH - ORIG_WIDTH) // 2)
                x = pad(x, (height_padding, width_padding), mode='reflect')
                X_samples[n] = np.reshape(x, newshape=(im_height, im_width, IM_CHAN))
            else:
                X_samples[n] = np.reshape(x, newshape=(im_height, im_width, IM_CHAN))

        print('Done!')

        id_samples = np.array(test_ids)

        self.id_samples = id_samples
        self.X_samples = X_samples

    @property
    def num_samples(self):
        return len(self.X_samples)

    @property
    def num_train(self):
        return len(self.X_train)

    @property
    def num_valid(self):
        return len(self.X_valid)

    def kfold_split(self, n_splits, idx_kfold, random_state=None):
        assert n_splits > idx_kfold
        kf = KFold(n_splits)
        for idx, (train_index, valid_index) in enumerate(kf.split(range(self.num_samples))):
            if idx == idx_kfold:
                break

        self.X_train = self.X_samples[train_index]
        self.Y_train = self.Y_samples[train_index]
        self.id_train = self.id_samples[train_index]

        self.X_valid = self.X_samples[valid_index]
        self.Y_valid = self.Y_samples[valid_index]
        self.id_valid = self.id_samples[valid_index]

    def create_train_generator(self, n_splits=10, idx_kfold=0, batch_size=8, augment_dict={}, shuffle=True, with_id=False):
        self.kfold_split(n_splits, idx_kfold)
        data_gen_args = augment_dict
        print("data_gen_args is {}".format(data_gen_args))
        seed = 1


        X_train_datagen = ImageDataGenerator(**data_gen_args)
        Y_train_datagen = ImageDataGenerator(**data_gen_args)
        id_train = self.id_train if with_id else None
        X_train_generator = X_train_datagen.flow(self.X_train, y=id_train, seed=seed, batch_size=batch_size, shuffle=shuffle)
        Y_train_generator = Y_train_datagen.flow(self.Y_train, seed=seed, batch_size=batch_size, shuffle=shuffle)
        train_generator = zip(X_train_generator, Y_train_generator)

        X_valid_datagen = ImageDataGenerator()
        Y_valid_datagen = ImageDataGenerator()
        id_valid = self.id_valid if with_id else None
        X_valid_generator = X_valid_datagen.flow(self.X_valid, y=id_valid, batch_size=batch_size, shuffle=False)
        Y_valid_generator = Y_valid_datagen.flow(self.Y_valid, batch_size=batch_size, shuffle=False)
        valid_generator = zip(X_valid_generator, Y_valid_generator)

        return train_generator, valid_generator

    def create_test_generator(self, batch_size=8, shuffle=False, with_id=False):
        seed = 1
        X_sample_datagen = ImageDataGenerator()
        id_samples = self.id_samples if with_id else None
        X_sample_generator = X_sample_datagen.flow(self.X_samples, y=id_samples, seed=seed, batch_size=batch_size, shuffle=shuffle)
        return X_sample_generator


def input_test(path_test):
    test_ids = next(os.walk(os.path.join(path_test, "images")))[2]

    X_test = np.zeros((len(test_ids), IM_HEIGHT, IM_WIDTH, IM_CHAN), dtype=np.uint8)
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

