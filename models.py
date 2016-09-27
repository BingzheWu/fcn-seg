import tensorflow as tf
import numpy as np
import tflearn

def alexnet_fcn_voc(data_input):
    '''
    data_input is the input placeholder.
    return: inference of images which has the same dim of the input.
    '''
    #net = input_data
    pad = [[0,0], [50,50], [50, 50], [0,0]]
    data_input_with_pad = tf.pad(data_input, paddings = pad)
    conv1 =  tflearn.conv_2d(data_input_with_pad, nb_filter = 96, filter_size = 11, strides = 4, regularizer = 'L2', weight_decay = 0.001, padding = 'valid', activation = 'relu')
    pool1 = tflearn.max_pool_2d(conv1, kernel_size = 3, strides = 2 , padding = 'valid')
    conv2 = tflearn.conv_2d(pool1, nb_filter = 256, filter_size = 5, strides =1, regularizer = 'L2', activation = 'relu' )
    pool2 = tflearn.max_pool_2d(conv2, kernel_size = 3, strides = 2, padding = 'valid')
    conv3 = tflearn.conv_2d(conv2, nb_filter = 384, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu')
    conv4 = tflearn.conv_2d(conv3, nb_filter = 384, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu')
    conv5 = tflearn.conv_2d(conv4, nb_filter = 256, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu')
    pool5 = tflearn.conv_2d(conv5, kernel_size = 3, strides = 2, padding = 'valid')
    fc6 = tflearn.conv_2d(pool5,nb_filter = 4096,  filter_size = 6, strides = 1, padding = 'valid', activation = 'relu')
    fc7 = tflearn.conv_2d(fc6, nb_filter = 4096, filter_size = 1, strides = 1, padding = 'valid', activation = 'relu')
    up_score = tflearn.layer.conv.upscore_layer(fc7, num_classes = 21, kernel_size = 63, strides = 32 )
    return up_score

