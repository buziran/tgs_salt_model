# -*- coding: utf-8 -*-

import tensorflow as tf
import config

tf.flags.DEFINE_string(
    'log', './output/log', """path to log directory""")

tf.flags.DEFINE_integer(
    'epochs', 300, """path to log directory""")

tf.flags.DEFINE_integer(
    'batch_size', 32, """batch size""")

tf.flags.DEFINE_bool(
    'early_stopping', False, help="""whether to apply early-stopping""")

tf.flags.DEFINE_float(
    'lr', 0.001, help="""initial value of learning rate""")

tf.flags.DEFINE_float(
    'lr_decay', 1.0, help="""decay factor for learning rate""")

tf.flags.DEFINE_integer(
    'epochs_decay', 10, help="""decay epoch of learning rate""")

"""Model"""
tf.flags.DEFINE_bool('batch_norm', False, """whether to use batch-normalization""")

tf.flags.DEFINE_float('drop_out', 0.0, """whether to use drop-out""")


"""Augmentations"""
tf.flags.DEFINE_bool(
    'legacy', False, """whether to use legacy code""")

tf.flags.DEFINE_bool(
    'horizontal_flip', True, """whether to apply horizontal flip""")

tf.flags.DEFINE_bool(
    'vertical_flip', True, """whether to apply vertical flip""")

tf.flags.DEFINE_integer(
    'rotation_range', 30, """random rotation range""")

tf.flags.DEFINE_float(
    'zoom_range', 0.2, """random zoom range""")

tf.flags.DEFINE_float(
    'shift_range', 0.0, """random shift range""")

tf.flags.DEFINE_enum(
    'fill_mode', 'reflect', enum_values=['constant', 'nearest', 'reflect', 'wrap'], help="""fill mode""")


