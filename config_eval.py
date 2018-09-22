# -*- coding: utf-8 -*-

import tensorflow as tf
import config

tf.flags.DEFINE_float(
    'weight_fg', 1.0, """weight of foreground mask""")

tf.flags.DEFINE_float(
    'weight_bg', 1.0, """weight of background mask""")

tf.flags.DEFINE_list(
    'weight_ad', None, """threshold of foreground coverage to apply weight ex) --weight_ad=0.1,0.5""")

tf.flags.DEFINE_bool(
    'best_threshold', True, """whether to search best threshold""")
