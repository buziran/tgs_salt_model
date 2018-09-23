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

def pad(image, target_shape, mode='CONSTANT', set_shape=True):
    target_height, target_width = target_shape
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    top = tf.cast((target_height - height) / 2, tf.int32)
    bottom = target_height - height - top
    left = tf.cast((target_width - width) / 2, tf.int32)
    right = target_width - width - left
    image = tf.pad(image, mode=mode, paddings=[[top, bottom], [left, right], [0, 0]])
    if set_shape:
        _, _, channels = image.get_shape().as_list()
        image.set_shape(shape=(target_height, target_width, channels))
    return image


def rotate(image, angle, interpolation='NEARESET'):
    image = tf.expand_dims(image, axis=0)
    image = tf.contrib.image.rotate(image, angle, interpolation)
    image = tf.squeeze(image, axis=0)
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

    def gen_test(self, adjust='resize', batch_size=32, repeat=1, with_path=True):

        paths_test_x = [os.path.join(self.path_input, 'images', idx) for idx in self.id_samples]

        dataset_test = tf.data.Dataset.from_tensor_slices(paths_test_x)

        if with_path:
            def _load_normalize(path_image):
                image = load_img(path_image, channels=IM_CHAN)
                return normalize(image), path_image

            def _adjust(image, path_image):
                if adjust == 'resize':
                    image = resize(image, target_shape=(IM_HEIGHT, IM_WIDTH), method=tf.image.ResizeMethod.BILINEAR)
                elif adjust in ['reflect', 'constant', 'symmetric']:
                    image = pad(image, target_shape=(IM_HEIGHT, IM_WIDTH), mode=adjust)
                else:
                    raise ValueError("adjust-mode {} is not supported".format(adjust))
                return image, path_image
        else:
            def _load_normalize(path_image):
                image = load_img(path_image, channels=IM_CHAN)
                return normalize(image)

            def _adjust(image):
                if adjust == 'resize':
                    image = resize(image, target_shape=(IM_HEIGHT, IM_WIDTH), method=tf.image.ResizeMethod.BILINEAR)
                elif adjust in ['reflect', 'constant', 'symmetric']:
                    image = pad(image, target_shape=(IM_HEIGHT, IM_WIDTH), mode=adjust)
                return image

        dataset_test = dataset_test.map(_load_normalize, num_parallel_calls=8)
        dataset_test = dataset_test.map(_adjust, num_parallel_calls=8)
        dataset_test = dataset_test.repeat(repeat)
        dataset_test = dataset_test.batch(batch_size)

        iter_test = dataset_test.make_one_shot_iterator()
        return iter_test

    def gen_train_valid(self, n_splits, idx_kfold,
                        adjust='resize', weight_fg=1.0, weight_bg=1.0, weight_adaptive=None,
                        batch_size=32, augment_dict=None, repeat=None):
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
            elif adjust in ['reflect', 'constant', 'symmetric']:
                image = pad(image, target_shape=(IM_HEIGHT, IM_WIDTH), mode=adjust)
                mask = pad(mask, target_shape=(IM_HEIGHT, IM_WIDTH), mode='CONSTANT')
                weight = pad(weight, target_shape=(IM_HEIGHT, IM_WIDTH), mode='CONSTANT')
            else:
                raise ValueError("adjust-mode {} is not supported".format(adjust))
            return image, mask, weight

        def _rand_flip(image, flip_fn, p):
            return tf.cond(p>0.5, true_fn=lambda: flip_fn(image), false_fn=lambda: image)

        def _rand_shift(image, mask, weight, height_shift_range, width_shift_range, mode='CONSTANT'):
            orig_height, orig_width, orig_channels = image.get_shape().as_list()
            height_shift_range = height_shift_range if height_shift_range is not None else 0.0
            width_shift_range = width_shift_range if width_shift_range is not None else 0.0
            target_height = tf.cast(IM_HEIGHT * (1+height_shift_range), dtype=tf.int32)
            target_width = tf.cast(IM_WIDTH * (1+width_shift_range), dtype=tf.int32)
            image = pad(image, target_shape=(target_height, target_width), mode=mode, set_shape=False)
            mask = pad(mask, target_shape=(target_height, target_width), mode='CONSTANT', set_shape=False)
            weight = pad(weight, target_shape=(target_height, target_width), mode='CONSTANT', set_shape=False)
            image, mask, weight = _rand_crop(image, mask, weight, orig_height, orig_width)
            return image, mask, weight

        def _rand_erase(
                image, mask, weight, range_image, range_mask, range_weight,
                probability=0.5, min_size=0.02, max_size=0.4,
                min_aspect_ratio=0.3, max_aspect_ratio=1/0.3, pixel_wise=False, seed=None):
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

        def _mixup(images, masks):
            alpha = augment_dict['mixup']
            dist_beta = tf.distributions.Beta(alpha, alpha)
            lam = dist_beta.sample((1,1,1,1))
            mixup_factor = tf.concat([lam, 1-lam], axis=0)
            image = tf.reduce_sum(images * mixup_factor, axis=0, keepdims=False)
            mask = tf.reduce_sum(masks * mixup_factor, axis=0, keepdims=False)
            return image, mask

        def _rand_crop(image, mask, weight, target_height, target_width):
            orig_shape = tf.shape(image)
            orig_height, orig_width = orig_shape[0], orig_shape[1]
            offset_height = tf.cond(tf.equal(0, orig_height-target_height),
                                    true_fn=lambda:0,
                                    false_fn=lambda:tf.random_uniform((), 0, orig_height-target_height, dtype=tf.int32))
            offset_width = tf.cond(tf.equal(0, orig_width-target_width),
                                   true_fn=lambda:0,
                                   false_fn=lambda:tf.random_uniform((), 0, orig_width-target_width, dtype=tf.int32))
            image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
            mask = tf.image.crop_to_bounding_box(mask, offset_height, offset_width, target_height, target_width)
            weight = tf.image.crop_to_bounding_box(weight, offset_height, offset_width, target_height, target_width)
            return image, mask, weight

        def _pad(image, mask, weight, target_height, target_width, mode):
            image = pad(image, (target_height, target_width), mode=mode)
            mask = pad(mask, (target_height, target_width), mode='CONSTANT')
            weight = pad(weight, (target_height, target_width), mode='CONSTANT')
            image.set_shape((target_height, target_width, IM_CHAN))
            mask.set_shape((target_height, target_width, 1))
            weight.set_shape((target_height, target_width, 1))
            return image, mask, weight

        def _augment(image, mask, weight):
            if augment_dict is None:
                return image, mask, weight
            mode = augment_dict['fill_mode']
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
            if augment_dict['brightness_range'] is not None:
                max_delta = augment_dict['brightness_range']
                image = tf.image.random_brightness(image, max_delta)
            if augment_dict['zoom_range'] is not None and augment_dict['zoom_range'] != 0.0:
                zoom_range = augment_dict['zoom_range']
                zoom = tf.random_uniform((), (1-zoom_range), (1+zoom_range), dtype=tf.float32)
                target_height = tf.cast(IM_HEIGHT * zoom, dtype=tf.int32)
                target_width = tf.cast(IM_WIDTH * zoom, dtype=tf.int32)
                image = tf.image.resize_images(image, size=(target_height, target_width))
                mask = tf.image.resize_images(mask, size=(target_height, target_width))
                weight = tf.image.resize_images(weight, size=(target_height, target_width))
                image, mask, weight = tf.cond(zoom>1.0,
                                              true_fn=lambda:_rand_crop(image, mask, weight, IM_HEIGHT, IM_WIDTH),
                                              false_fn=lambda:_pad(image, mask, weight, IM_HEIGHT, IM_WIDTH, mode=mode), strict=True)
            if augment_dict['rotation_range'] is not None:
                rot = augment_dict['rotation_range'] * np.math.pi / 180
                angle = tf.random_uniform((), -rot, rot, dtype=tf.float32)
                interp = 'BILINEAR'
                image = rotate(image, angle, interp)
                mask = rotate(mask, angle, interp)
                weight = rotate(weight, angle, interp)
            if augment_dict['height_shift_range'] is not None or augment_dict['width_shift_range'] is not None:
                image, mask, weight = _rand_shift(
                    image, mask, weight, augment_dict['height_shift_range'], augment_dict['width_shift_range'], mode=mode)
            if augment_dict['random_erase'] is not None:
                if augment_dict['random_erase'] == 'constant':
                    pixel_wise = False
                    range_image = (0, 1)
                elif augment_dict['random_erase'] == 'zero':
                    pixel_wise = False
                    range_image = (0, 0)
                elif augment_dict['random_erase'] == 'pixel':
                    pixel_wise = True
                    range_image = (0, 1)
                else:
                    raise NotImplementedError()
                image, mask, weight = _rand_erase(image, mask, weight,
                                   range_image=range_image, range_mask=(0, 0), range_weight=(0, 0), pixel_wise=pixel_wise)
            return image, mask, weight

        def _concat_mask_weight(image, mask, weight):
            mask_and_weight = tf.concat((mask, weight), axis=2)
            return image, mask_and_weight

        dataset_train = dataset_train.shuffle(batch_size*10)
        dataset_train = dataset_train.map(_load_normalize, num_parallel_calls=8)

        if augment_dict is not None and augment_dict['mixup'] is not None:
            dataset_train = dataset_train.batch(2)
            dataset_train = dataset_train.map(_mixup, num_parallel_calls=8)

        dataset_train = dataset_train.map(_create_weight, num_parallel_calls=8)
        dataset_train = dataset_train.map(_adjust, num_parallel_calls=8)
        dataset_train = dataset_train.map(_augment, num_parallel_calls=8)
        dataset_train = dataset_train.map(_concat_mask_weight)
        dataset_train = dataset_train.repeat(repeat)
        dataset_train = dataset_train.batch(batch_size)
        dataset_train = dataset_train.prefetch(1)

        dataset_valid = dataset_valid.map(_load_normalize, num_parallel_calls=8)
        dataset_valid = dataset_valid.map(_create_weight, num_parallel_calls=8)
        dataset_valid = dataset_valid.map(_adjust, num_parallel_calls=8)
        dataset_valid = dataset_valid.map(_concat_mask_weight)
        dataset_valid = dataset_valid.repeat(repeat)
        dataset_valid = dataset_valid.batch(batch_size)
        dataset_valid = dataset_valid.prefetch(1)

        if repeat is None:
            iter_train = dataset_train.make_one_shot_iterator()
            iter_valid = dataset_valid.make_one_shot_iterator()
        else:
            iter_train = dataset_train.make_initializable_iterator()
            iter_valid = dataset_valid.make_initializable_iterator()
        return iter_train, iter_valid


