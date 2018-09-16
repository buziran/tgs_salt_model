import os
import sys

import cv2
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
import tensorflow as tf

from constant import *
from random_erase import RandomErasing


def load_img(filename, channels=3):
    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string, channels=channels)
    return image

def normalize(image):
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.
    return image

def resize(image, target_shape, method=tf.image.ResizeMethod.BILINEAR):
    image = tf.image.resize_images(image, target_shape, method=method)
    return image

def pad(image, target_shape, mode='CONSTANT', constant_values=0):
    target_height, target_width = target_shape
    height, width, channels = image.get_shape()
    top = tf.cast((target_height - height) / 2, tf.int32)
    bottom = target_height - height - top
    left = tf.cast((target_width - width) / 2, tf.int32)
    right = target_width - width - left
    image = tf.pad(image, mode=mode, paddings=[[top, bottom], [left, right], [0, 0]])
    return image

class Dataset(object):
    def __init__(self, path_input):
        self.path_input = path_input
        id_samples = next(os.walk(os.path.join(self.path_input, "images")))[2]
        id_samples = sorted(id_samples)
        self.id_samples = id_samples

    def __len__(self):
        return len(self.id_samples)

    def kfold_split(self, n_splits, idx_kfold):
        assert n_splits > idx_kfold
        kf = KFold(n_splits)
        train_index = []
        valid_index = []
        for idx, (_train_index, _valid_index) in enumerate(kf.split(range(len(self)))):
            if idx == idx_kfold:
                train_index = _train_index
                valid_index = _valid_index
                break
        id_train = np.asarray(self.id_samples)[train_index]
        id_valid = np.asarray(self.id_samples)[valid_index]
        return id_train, id_valid

    def len_train_valid(self, n_splits, idx_kfold):
        id_train, id_valid = self.kfold_split(n_splits, idx_kfold)
        return len(id_train), len(id_valid)

    def gen_train_valid(self, n_splits, idx_kfold,
                        adjust='resize', weight_fg=1.0, weight_bg=1.0, weight_adaptive=None,
                        batch_size=32, augment_dict=None):
        id_train, id_valid = self.kfold_split(n_splits, idx_kfold)

        paths_train_x = [os.path.join(self.path_input, 'images', idx) for idx in id_train]
        paths_train_y = [os.path.join(self.path_input, 'masks', idx) for idx in id_train]
        paths_valid_x = [os.path.join(self.path_input, 'images', idx) for idx in id_valid]
        paths_valid_y = [os.path.join(self.path_input, 'masks', idx) for idx in id_valid]

        dataset_train_x = tf.data.Dataset.from_tensor_slices(paths_train_x)
        dataset_train_y = tf.data.Dataset.from_tensor_slices(paths_train_y)
        dataset_valid_x = tf.data.Dataset.from_tensor_slices(paths_valid_x)
        dataset_valid_y = tf.data.Dataset.from_tensor_slices(paths_valid_y)
        dataset_train  = tf.data.Dataset.zip((dataset_train_x, dataset_train_y))
        dataset_valid  = tf.data.Dataset.zip((dataset_valid_x, dataset_valid_y))

        def _load_normalize(path_image, path_mask):
            image = load_img(path_image, channels=IM_CHAN)
            mask = load_img(path_mask, channels=1)
            return normalize(image), normalize(mask)

        def _create_weight(image, mask):
            weight = tf.ones_like(mask, dtype=tf.float32)
            if weight_fg == 1.0 and weight_bg == 1.0 and weight_adaptive is None:
                pass
            elif weight_adaptive is None:
                fg = tf.ones_like(mask) * weight_fg
                bg = tf.ones_like(mask) * weight_bg
                weight = tf.where(mask > 0.5, fg, bg)
            elif weight_adaptive is not None:
                raise NotImplementedError()
            return image, mask, weight

        def _adjust(image, mask, weight):
            if adjust == 'resize':
                image = resize(image, target_shape=(IM_HEIGHT, IM_WIDTH), method=tf.image.ResizeMethod.BILINEAR)
                mask = resize(mask, target_shape=(IM_HEIGHT, IM_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                weight = resize(weight, target_shape=(IM_HEIGHT, IM_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            elif adjust == 'pad':
                image = pad(image, target_shape=(IM_HEIGHT, IM_WIDTH), mode='CONSTANT')
                mask = pad(mask, target_shape=(IM_HEIGHT, IM_WIDTH), mode='CONSTANT')
                weight = pad(weight, target_shape=(IM_HEIGHT, IM_WIDTH), mode='CONSTANT')
            return image, mask, weight

        def _rand_flip(image, flip_fn, p):
            return tf.cond(p>0.5, true_fn=lambda: flip_fn(image), false_fn=lambda: image)

        def _rand_shift(image, height_shift_range, width_shift_range, seed=None):
            orig_height, orig_width, orig_channels = image.get_shape()
            height_shift_range = height_shift_range if height_shift_range is not None else 0.0
            width_shift_range = width_shift_range if width_shift_range is not None else 0.0
            target_height = tf.cast(IM_HEIGHT * (1+height_shift_range), dtype=tf.int32)
            target_width = tf.cast(IM_WIDTH * (1+width_shift_range), dtype=tf.int32)
            image = pad(image, target_shape=(target_height, target_width), mode='CONSTANT')
            image = tf.random_crop(image, size=(orig_height, orig_width, orig_channels), seed=seed)
            return image

        def _rand_erase(
                image, mask, weight, range_image, range_mask, range_weight,
                probability=0.5, min_size=0.02, max_size=0.4,
                min_aspect_ratio=0.3, max_aspect_ratio=1/0.3, min_val=0, max_val=1, pixel_wise=False, seed=None):
            # Generate seed for reproductivity
            if seed is not None:
                np.random.seed(seed)
                seeds = np.random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=[5])
                seed_size, seed_ratio, seed_left, seed_top, seed_prob = seeds
            else:
                seed_size, seed_ratio, seed_left, seed_top, seed_prob = [None] * 5
            height, width, channels = image.get_shape().as_list()
            num_elems = tf.cast(height, dtype=tf.float32) * tf.cast(width, dtype=tf.float32)
            s = tf.random_uniform((), min_size, max_size, seed=seed_size) * num_elems
            log_min_asp = tf.log(min_aspect_ratio)
            log_max_asp = tf.log(max_aspect_ratio)
            r = tf.exp(tf.random_uniform((), log_min_asp, log_max_asp, seed=seed_ratio))
            w = tf.cast(tf.sqrt(s / r), dtype=tf.int32)
            h = tf.cast(tf.sqrt(s * r), dtype=tf.int32)
            w = tf.reduce_min([width, w])
            h = tf.reduce_min([height, h])
            left = tf.cond(tf.equal(w, width),
                lambda:0,
                lambda:tf.random_uniform((), 0, width-w, seed=seed_left, dtype=tf.int32))
            top = tf.cond(tf.equal(h, height),
                lambda:0,
                lambda:tf.random_uniform((), 0, height-h, seed=seed_top, dtype=tf.int32))

            erased_image = _rand_bbox(image, h, w, top, left, min_val=range_image[0], max_val=range_image[1],
                                      pixel_wise=pixel_wise)
            erased_mask = _rand_bbox(mask, h, w, top, left, min_val=range_mask[0], max_val=range_mask[1],
                                     pixel_wise=False)
            erased_weight = _rand_bbox(weight, h, w, top, left, min_val=range_weight[0], max_val=range_weight[1],
                                       pixel_wise=False)

            prob = tf.random_uniform((), seed=seed_prob)
            image = tf.cond(tf.less(prob, probability), true_fn=lambda:erased_image, false_fn=lambda:image)
            mask = tf.cond(tf.less(prob, probability), true_fn=lambda:erased_mask, false_fn=lambda:mask)
            weight = tf.cond(tf.less(prob, probability), true_fn=lambda:erased_weight, false_fn=lambda:weight)
            return image, mask, weight

        def _rand_bbox(image, height, width, top, left, min_val, max_val, pixel_wise):
            im_height, im_width, im_ch = image.get_shape().as_list()
            paddings = ([[top, im_height-top-height], [left, im_width-left-width], [0, 0]])
            randomize_mask = tf.pad(tf.ones(shape=(height, width, im_ch), dtype=tf.bool), paddings=paddings)
            if pixel_wise:
                values = tf.random_uniform((height, width, im_ch), min_val, max_val, dtype=image.dtype)
            else:
                _val = tf.random_uniform((), min_val, max_val, dtype=image.dtype)
                values = tf.fill(dims=(im_height, im_width, im_ch), value=_val)
            randomized = tf.where(randomize_mask, values, image)
            return randomized

        def _augment(image, mask, weight):
            if augment_dict is None:
                return image, mask, weight
            if augment_dict['horizontal_flip']:
                p = tf.random_uniform(())
                image = _rand_flip(image, tf.image.flip_left_right, p)
                mask = _rand_flip(mask, tf.image.flip_left_right, p)
                weight = _rand_flip(weight, tf.image.flip_left_right, p)
            if augment_dict['vertical_flip']:
                p = tf.random_uniform(())
                image = _rand_flip(image, tf.image.flip_up_down, p)
                mask = _rand_flip(mask, tf.image.flip_up_down, p)
                weight = _rand_flip(weight, tf.image.flip_up_down, p)
            if augment_dict['zoom_range'] is not None:
                zoom_range = augment_dict['zoom_range']
                zoom = tf.random_uniform((), (1-zoom_range), (1+zoom_range), dtype=tf.float32)
                target_height = tf.cast(IM_HEIGHT * zoom, dtype=tf.int32)
                target_width = tf.cast(IM_WIDTH * zoom, dtype=tf.int32)
                image = tf.image.resize_images(image, size=(target_height, target_width))
                mask = tf.image.resize_images(mask, size=(target_height, target_width))
                weight = tf.image.resize_images(weight, size=(target_height, target_width))
                image = tf.image.resize_image_with_crop_or_pad(image, IM_HEIGHT, IM_WIDTH)
                mask = tf.image.resize_image_with_crop_or_pad(mask, IM_HEIGHT, IM_WIDTH)
                weight = tf.image.resize_image_with_crop_or_pad(weight, IM_HEIGHT, IM_WIDTH)
            if augment_dict['height_shift_range'] is not None or augment_dict['width_shift_range'] is not None:
                image = _rand_shift(
                    image, augment_dict['height_shift_range'], augment_dict['width_shift_range'], seed=17)
                mask = _rand_shift(
                    mask, augment_dict['height_shift_range'], augment_dict['width_shift_range'], seed=17)
                weight = _rand_shift(
                    weight, augment_dict['height_shift_range'], augment_dict['width_shift_range'], seed=17)
            if augment_dict['random_erase'] is not None:
                if augment_dict['random_erase'] == 'constant':
                    pixel_wise = False
                elif augment_dict['random_erase'] == 'pixel':
                    pixel_wise = True
                else:
                    raise NotImplementedError()
                image, mask, weight = _rand_erase(image, mask, weight,
                                   range_image=(0, 1), range_mask=(0, 0), range_weight=(0, 0), pixel_wise=pixel_wise, probability=1)
            return image, mask, weight

        def _concat_mask_weight(image, mask, weight):
            mask_and_weight = tf.concat((mask, weight), axis=2)
            return image, mask_and_weight

        dataset_train = dataset_train.shuffle(batch_size*10)
        dataset_train = dataset_train.map(_load_normalize, num_parallel_calls=8)
        dataset_train = dataset_train.map(_create_weight, num_parallel_calls=8)
        dataset_train = dataset_train.map(_adjust, num_parallel_calls=8)
        dataset_train = dataset_train.map(_augment, num_parallel_calls=8)
        dataset_train = dataset_train.map(_concat_mask_weight)
        dataset_train = dataset_train.repeat()
        dataset_train = dataset_train.batch(batch_size)

        dataset_valid = dataset_valid.map(_load_normalize, num_parallel_calls=8)
        dataset_valid = dataset_valid.map(_create_weight, num_parallel_calls=8)
        dataset_valid = dataset_valid.map(_adjust, num_parallel_calls=8)
        dataset_valid = dataset_valid.map(_concat_mask_weight)
        dataset_valid = dataset_valid.repeat()
        dataset_valid = dataset_valid.batch(batch_size)

        iter_train = dataset_train.make_one_shot_iterator()
        iter_valid = dataset_valid.make_one_shot_iterator()
        return iter_train, iter_valid


