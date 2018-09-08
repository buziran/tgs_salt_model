# -*- coding: utf-8 -*-

import tensorflow as tf
import config

tf.flags.DEFINE_enum(
    'adjust', 'resize', enum_values=['resize', 'resize-cv', 'pad'], help="""mode to adjust image size""")
