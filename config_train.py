# -*- coding: utf-8 -*-

import tensorflow as tf
import config

tf.flags.DEFINE_string(
    'log', '../output/log', """path to log directory""")

tf.flags.DEFINE_integer(
    'epochs', 300, """path to log directory""")

tf.flags.DEFINE_bool(
    'early_stopping', False, help="""whether to apply early-stopping""")

tf.flags.DEFINE_string('restore', None, """path to model directory to restore""")

tf.flags.DEFINE_enum(
    'opt', 'adam',
    enum_values=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'], help="""optimizer""")

tf.flags.DEFINE_float(
    'lr', 0.001, help="""initial value of learning rate""")

tf.flags.DEFINE_float(
    'lr_decay', 1.0, help="""decay factor for learning rate""")

tf.flags.DEFINE_float(
    'weight_decay', 0.0, help="""weight decay""")

tf.flags.DEFINE_bool(
    'exclude_bn', True, help="""exclude parameter of batch normalization from l2 loss""")

tf.flags.DEFINE_bool(
    'freeze_once', False, """whether to freeze learning rate once""")

tf.flags.DEFINE_integer(
    'epochs_decay', 10, help="""decay epoch of learning rate""")

tf.flags.DEFINE_float(
    'weight_fg', 1.0, """weight of foreground mask""")

tf.flags.DEFINE_float(
    'weight_bg', 1.0, """weight of background mask""")

tf.flags.DEFINE_list(
    'weight_ad', None, """threshold of foreground coverage to apply weight ex) --weight_ad=0.1,0.5""")

tf.flags.DEFINE_bool(
    'debug', False, "Run as debug mode")

"""Model"""
tf.flags.DEFINE_enum(
    'pretrained', None, enum_values=['resnet50', 'inception_resnet_v2', 'densenet121'],
    help="""whether to use batch-normalization""")

tf.flags.DEFINE_enum(
    'contrib', None, enum_values=['resnet18', 'resnet34', 'resnet50'],
    help="""contribution model of keras-contrib""")

tf.flags.DEFINE_float('spatial_dropout', None, """factor of spatial dropout""")

tf.flags.DEFINE_bool('batch_norm', True, """whether to use batch-normalization""")

tf.flags.DEFINE_float('drop_out', 0.0, """whether to use drop-out""")

tf.flags.DEFINE_bool('use_ref', True, """whether to use reference model""")

tf.flags.DEFINE_integer('depth', 5, """number of channel at UNet first layer""")

tf.flags.DEFINE_integer('start_ch', 16, """number of channel at UNet first layer""")

"""Augmentations"""
tf.flags.DEFINE_bool(
    'horizontal_flip', True, """whether to apply horizontal flip""")

tf.flags.DEFINE_bool(
    'vertical_flip', True, """whether to apply vertical flip""")

tf.flags.DEFINE_integer(
    'rotation_range', 0, """random rotation range""")

tf.flags.DEFINE_float(
    'zoom_range', 0.2, """random zoom range""")

tf.flags.DEFINE_float(
    'shift_range', 0.0, """random shift range""")

tf.flags.DEFINE_enum(
    'fill_mode', 'reflect', enum_values=['constant', 'nearest', 'reflect', 'wrap'], help="""fill mode""")

tf.flags.DEFINE_enum(
    'random_erase', 'constant', enum_values=['pixel', 'constant', 'zero'], help="""mode to fill in random-erasing""")

tf.flags.DEFINE_float(
    'mixup', None, help="""alpha value of mixup""")

