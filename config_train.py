# -*- coding: utf-8 -*-

import tensorflow as tf
import config

tf.flags.DEFINE_string(
    'log', '../output/log', """path to log directory""")

tf.flags.DEFINE_integer(
    'epochs', 300, """path to log directory""")

tf.flags.DEFINE_string('restore', None, """path to model directory to restore""")

tf.flags.DEFINE_string('restore_weight', None, """path to weight directory to restore""")

tf.flags.DEFINE_float(
    'weight_fg', 1.0, """weight of foreground mask""")

tf.flags.DEFINE_float(
    'weight_bg', 1.0, """weight of background mask""")

tf.flags.DEFINE_list(
    'weight_ad', None, """threshold of foreground coverage to apply weight ex) --weight_ad=0.1,0.5""")

tf.flags.DEFINE_bool(
    'debug', False, "Run as debug mode")

"""Optimize"""

tf.flags.DEFINE_enum(
    'opt', 'adam',
    enum_values=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam', 'msgd'], help="""optimizer""")

tf.flags.DEFINE_float(
    'lr', 0.001, help="""initial value of learning rate""")

tf.flags.DEFINE_float(
    'lr_decay', 1.0, help="""decay factor for learning rate""")

tf.flags.DEFINE_string(
    'epochs_decay', '10', help="""decay epoch of learning rate""")

tf.flags.DEFINE_bool(
    'cyclic', False, help="""whether to use cyclic learning rate""")

tf.flags.DEFINE_integer(
    'epoch_size', None, help="""[cyclic lr] epoch-size for cyclic learning rate""")

tf.flags.DEFINE_float(
    'max_lr', None, help="""[cyclic lr] max learning rate for cyclic learning rate""")

tf.flags.DEFINE_enum(
    'mode_clr', "triangular2", enum_values=['triangular', 'triangular2'],
    help="""[cyclic lr] max learning rate for cyclic learning rate""")

tf.flags.DEFINE_float(
    'weight_decay', 0.0, help="""weight decay""")

tf.flags.DEFINE_bool(
    'exclude_bn', True, help="""exclude parameter of batch normalization from l2 loss""")

tf.flags.DEFINE_bool(
    'freeze_once', False, """whether to freeze learning rate once""")

tf.flags.DEFINE_bool(
    'early_stopping', False, help="""whether to apply early-stopping""")

tf.flags.DEFINE_bool(
    'reduce_on_plateau', False, help="""whether to reduce learning rate on plateau""")

tf.flags.DEFINE_bool(
    'save_best_only', True, help="""whether to save best score model or save latest model""")

"""Dataset"""

tf.flags.DEFINE_bool(
    "filter_vert_hori", True, "whether to filter vertical/horizontal mask")

tf.flags.DEFINE_float(
    "ignore_tiny", 0.0, "ignore mask when (fg ratio) < ignore_tiny")

"""Model"""
tf.flags.DEFINE_enum(
    'pretrained', None, enum_values=['resnet50', 'resnet50-shallow', 'inception_resnet_v2', 'densenet121'],
    help="""pretrained model of keras-applications""")

tf.flags.DEFINE_bool('renorm', None, help="""whether to use batch-renormalization""")

tf.flags.DEFINE_bool('retrain', True, """whether to retrain layers in pretrained model""")

tf.flags.DEFINE_enum(
    'contrib', None, enum_values=['resnet34', 'resnet50'],
    help="""contribution model of keras-contrib""")

tf.flags.DEFINE_enum(
    'residual_unit', 'v1', enum_values=['v1', 'v2'], help="which residual unit to use in contibuted resnet")

tf.flags.DEFINE_bool('preprocess', True, """whether to preprocess image to fit [-1, 1]""")

tf.flags.DEFINE_float('spatial_dropout', None, """factor of spatial dropout""")

tf.flags.DEFINE_bool('batch_norm', True, """whether to use batch-normalization""")

tf.flags.DEFINE_float('drop_out', 0.0, """whether to use drop-out""")

tf.flags.DEFINE_bool('use_ref', True, """whether to use reference model""")

tf.flags.DEFINE_integer('depth', 5, """number of channel at UNet first layer""")

tf.flags.DEFINE_integer('start_ch', 16, """number of channel at UNet first layer""")

tf.flags.DEFINE_integer('last_kernel', 1, """kernel size of last convolution layer""")

tf.flags.DEFINE_bool('last_1x1', False, """whether to add 1x1 conv as last layer""")

"""Augmentations"""
tf.flags.DEFINE_bool(
    'augment', True, """whether to apply augment""")

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

tf.flags.DEFINE_float(
    'brightness_range', 0.0, """random brightness range""")

tf.flags.DEFINE_float(
    'gradation_range', 0.0, """random gradation range""")

tf.flags.DEFINE_enum(
    'fill_mode', 'reflect', enum_values=['constant', 'nearest', 'reflect', 'wrap'], help="""fill mode""")

tf.flags.DEFINE_enum(
    'random_erase', 'constant', enum_values=['pixel', 'constant', 'zero', 'none'], help="""mode to fill in random-erasing""")

tf.flags.DEFINE_float(
    'mixup', None, help="""alpha value of mixup""")

