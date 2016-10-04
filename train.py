import tflearn
import tensorflow as tf
from models import alexnet_fcn_voc, fcn8, res
import numpy as np
from data_utils.voc_utils import load_data
import config
import time
FLAGS = tf.app.flags.FLAGS
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, cross_entropy_mean)
    #return cross_entropy_mean
    return tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name = 'total_loss')
def train_op(total_loss, global_step):
    lr = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_steps, FLAGS.decay_factor, staircase = True)
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step = global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op_ = tf.no_op(name = 'train')
    return train_op_
def input_():
    img_prep = tflearn.ImagePreprocessing()
    img_prep.add_featurewise_zero_center( per_channel = True, mean =[104.00698, 116.66876762, 122.67891434])
    input_data = tflearn.input_data(shape = [None, None, None, 3], data_preprocessing = img_prep)
    return input_data
def init_config(log_device = False, gpu_memory_fraction = 0.5):
    gs = tf.GPUOptions( per_process_gpu_memory_fraction = gpu_memory_fraction)
    config_ = tf.ConfigProto(log_device_placement = log_device, gpu_options = gs)
    return config_
    
def train():
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable = False)
        label_ = tf.placeholder("float", [None])
        image_ = input_()
        shape_ = tf.placeholder("int32", [4])
        #logits = alexnet_fcn_voc(image_, shape_)
        logits = res(image_, shape_, 5)
        logits = tf.reshape(logits, (-1, FLAGS.num_class))
        loss_ = loss(logits, label_)
        train_op_ = train_op(loss_, global_step)
        sess = tf.Session(config = init_config())
        init = tf.initialize_all_variables()
        sess.run(init)
        for step in xrange(FLAGS.max_steps):
            data, label = load_data(step, 1)
            #image_.set_shape([None,None,None,3])
            #image_.set_shape(data.shape)
            logits_ = sess.run(logits, feed_dict = {image_: data, shape_: data.shape})
            loss_value, _,logits_tmp = sess.run([loss_,train_op_, tf.argmax(tf.nn.softmax(logits),dimension = 1)], feed_dict = {label_:label, image_:data, shape_:data.shape})
            print label
            print logits_tmp
            print loss_value
if __name__ == '__main__':
    train()

