import tensorflow as tf

tf.flags.DEFINE_string(
    'input', "../input/train", """path to train data""")

tf.flags.DEFINE_string(
    'model', '../output/model', """path to model directory""")

tf.flags.DEFINE_integer(
    'cv', 0, help="""index of k-fold cross validation. index must be in 0~9""")

tf.flags.DEFINE_bool('dice', True, """whether to use dice loss""")

