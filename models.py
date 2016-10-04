import tensorflow as tf
import numpy as np
import tflearn
import config
FLAGS = tf.app.flags.FLAGS
def alexnet_fcn_voc(data_input, shape_):
    '''
    data_input is the input placeholder.
    return: inference of images which has the same dim of the input.
    '''
    conv1 =  tflearn.conv_2d(data_input, nb_filter = 96, filter_size = 11, strides = 4, regularizer = 'L2', weight_decay = 0.001, padding = 'same', activation = 'relu')
    pool1 = tflearn.max_pool_2d(conv1, kernel_size = 3, strides = 2 , padding = 'same')
    conv2 = tflearn.conv_2d(pool1, nb_filter = 256, filter_size = 3, strides =1, regularizer = 'L2', activation = 'relu' )
    pool2 = tflearn.max_pool_2d(conv2, kernel_size = 3, strides = 2, padding = 'same')
    conv3 = tflearn.conv_2d(pool2, nb_filter = 384, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu')
    conv4 = tflearn.conv_2d(conv3, nb_filter = 384, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu')
    conv5 = tflearn.conv_2d(conv4, nb_filter = 256, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu')
    pool5 = tflearn.max_pool_2d(conv5, kernel_size = 3, strides = 2, padding = 'same')
    fc6 = tflearn.conv_2d(pool5,nb_filter = 4096,  filter_size = 6, strides = 1, padding = 'same', activation = 'relu')
    fc7 = tflearn.conv_2d(fc6, nb_filter = 4096, filter_size = 1, strides = 1, padding = 'same', activation = 'relu')
    fr = tflearn.conv_2d(fc7, nb_filter = 21, filter_size = 1, padding = 'same', activation = 'linear')
    up_score = tf.image.resize_bilinear(fr, size = [shape_[1], shape_[2]])
    #up_score = tflearn.layers.conv.upscore_layer(fr, num_classes = FLAGS.num_class , shape = shape_, kernel_size = 8, strides = 8 , trainable = False)
    #up_score = tflearn.layers.conv.upscore_layer(fr, num_classes = 21, kernel_size = 32, strides = 32 )
    #up_score_reshape = tf.reshape(up_score, (-1, 21))
    return up_score 
def fcn8(data_input, shape_):
    conv1_1 = tflearn.conv_2d(data_input, nb_filter = 64, filter_size = 3, strides = 1, regularizer = 'L2',activation = 'relu' )
    conv1_2 = tflearn.conv_2d(conv1_1, nb_filter = 64, filter_size = 3, strides = 1, regularizer = 'L2',activation = 'relu' )
    pool1 = tflearn.max_pool_2d(conv1_2, kernel_size = 3, strides = 2)
    conv2_1 = tflearn.conv_2d(pool1, nb_filter = 128, filter_size = 3, strides = 1,  regularizer = 'L2', activation = 'relu')
    conv2_2 = tflearn.conv_2d(conv2_1, nb_filter = 128, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu')
    pool2 = tflearn.max_pool_2d(conv2_2, kernel_size = 3, strides = 2 )
    conv3_1 = tflearn.conv_2d(pool2, nb_filter = 256, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu')
    conv3_2 = tflearn.conv_2d(conv3_1, nb_filter = 256, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu')
    conv3_3 = tflearn.conv_2d(conv3_2, nb_filter = 256, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu')
    pool3 = tflearn.max_pool_2d(conv3_3, kernel_size = 3, strides = 2)
    conv4_1 = tflearn.conv_2d(pool3, nb_filter = 512, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu' )
    conv4_2 = tflearn.conv_2d(conv4_1, nb_filter = 512, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu' )
    conv4_3 = tflearn.conv_2d(conv4_2, nb_filter = 512, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu' )
    pool4 = tflearn.max_pool_2d(conv4_3, kernel_size = 3, strides = 2)
    conv5_1 = tflearn.conv_2d(pool4, nb_filter = 512, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu' )
    conv5_2 = tflearn.conv_2d(conv5_1, nb_filter = 512, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu' )
    conv5_3 = tflearn.conv_2d(conv5_2, nb_filter = 512, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu' )
    pool5 = tflearn.max_pool_2d(conv5_3, kernel_size = 3, strides = 2)
    fc6 = tflearn.conv_2d(pool5, nb_filter = 250, filter_size = 7, activation = 'relu')
    fc7 = tflearn.conv_2d(fc6, nb_filter = 250, filter_size = 1, activation = 'relu')
    score_fr = tflearn.conv_2d(fc7, nb_filter = FLAGS.num_class, filter_size = 1)
    up_score = tf.image.resize_bilinear(score_fr, size = [shape_[1],shape_[2]])
    return up_score
def res(data_input, shape_, n):
    conv1 = tflearn.conv_2d(data_input, 32, 3, regularizer = 'L2', weight_decay = 0.0001, activation = 'relu', use_batch_norm = True)
    res1 = tflearn.residual_block(conv1, n, 32)
    res2 = tflearn.residual_block(res1, 1, 64, downsample = True )
    res3 = tflearn.residual_block(res2, n-1, 64)
    res4 = tflearn.residual_block(res3, 1, 128, downsample = True )
    res5 = tflearn.residual_block(res4, n-1, 128)
    res5_batch = tflearn.batch_normalization(res5)
    res5_batch = tflearn.activation(res5_batch, 'relu')
    fc = tflearn.conv_2d(res5_batch, nb_filter = 250, filter_size = 1, activation = 'relu')
    score_fr = tflearn.conv_2d(fc, nb_filter = FLAGS.num_class, filter_size = 1)
    up_score = tf.image.resize_bilinear(score_fr, size = [shape_[1], shape_[2]])
    return up_score

