import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float("learning_rate", 1e-5, "initial learning rate for training")
tf.app.flags.DEFINE_string("checkpoint_path", 'res_34_64/', "path to checkpoint_dir")
tf.app.flags.DEFINE_string('voc_dir', '/home/ceca/bingzhe/data/VOC2012', "This is /path/to/voc_data")
tf.app.flags.DEFINE_float('decay_factor', 0.01, "lr decay factor")
tf.app.flags.DEFINE_integer('decay_steps', 200, "lr decay step")
tf.app.flags.DEFINE_integer("max_steps", 2000, "the max step of train")
tf.app.flags.DEFINE_integer("num_class", 21, "the num of classes")
