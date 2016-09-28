import tflearn
import tensorflow as tf
from models import alexnet_fcn_voc
import numpy as np
from data_utils.voc_utils import load_data
import config
import time
FLAGS = tf.app.flags.FLAGS
def loss(logits, labels):
    logits = tf.reshape(logits, (-1, FLAGS.num_class))
    labels = tf.cast(labels, tf.int64)
    labels = tf.to_float(tf.reshape(labels, (-1)))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, cross_entropy_mean)
    return tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name = 'total_loss')
def train_op(total_loss, global_step):
    lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_steps, FLAGS.decay_factor, staircase = True)
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step = global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op_ = tf.no_op(name = 'train')
    return train_op_
def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable = False)
        label_ = tf.placeholder("float", [None, 21])
        image_ = tf.placeholder("float", [None, None, None, 3])
        shape_ = tf.placeholder("int32", [4])
        logits = alexnet_fcn_voc(image_, shape_)
        loss_ = loss(logits, label_)
        train_op_ = train_op(loss_, global_step)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        print tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        for step in xrange(FLAGS.max_steps):
            #label_.set_shape(label.shape)
            #image_.set_shape(data.shape)
            #print data.shape
            try:
                data, label = load_data(step, 1)
                print 1
                loss_value = sess.run(train_op_, feed_dict = {label_:label, image_:data, shape_:data.shape})
                print loss_value
            except:
                continue
if __name__ == '__main__':
    train()

