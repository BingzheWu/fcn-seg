import tensorflow as tf
import numpy as np
import tflearn

def alexnet_fcn_voc(data_input, shape_):
    '''
    data_input is the input placeholder.
    return: inference of images which has the same dim of the input.
    '''
    conv1 =  tflearn.conv_2d(data_input, nb_filter = 96, filter_size = 11, strides = 4, regularizer = 'L2', weight_decay = 0.001, padding = 'same', activation = 'relu')
    pool1 = tflearn.max_pool_2d(conv1, kernel_size = 3, strides = 2 , padding = 'same')
    conv2 = tflearn.conv_2d(pool1, nb_filter = 256, filter_size = 5, strides =1, regularizer = 'L2', activation = 'relu' )
    pool2 = tflearn.max_pool_2d(conv2, kernel_size = 3, strides = 2, padding = 'same')
    conv3 = tflearn.conv_2d(pool2, nb_filter = 384, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu')
    conv4 = tflearn.conv_2d(conv3, nb_filter = 384, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu')
    conv5 = tflearn.conv_2d(conv4, nb_filter = 256, filter_size = 3, strides = 1, regularizer = 'L2', activation = 'relu')
    pool5 = tflearn.max_pool_2d(conv5, kernel_size = 3, strides = 2, padding = 'same')
    fc6 = tflearn.conv_2d(pool5,nb_filter = 4096,  filter_size = 6, strides = 1, padding = 'same', activation = 'relu')
    fc7 = tflearn.conv_2d(fc6, nb_filter = 4096, filter_size = 1, strides = 1, padding = 'same', activation = 'relu')
    fr = tflearn.conv_2d(fc7, nb_filter = 21, filter_size = 1, padding = 'same', activation = 'linear')
    #shape = [1, 200, 769,3]
    up_score = tflearn.layers.conv.upscore_layer(fr, num_classes = 21, shape = shape_, kernel_size = 63, strides = 32 )
    #up_score = tflearn.layers.conv.upscore_layer(fr, num_classes = 21, kernel_size = 32, strides = 32 )
    #up_score_reshape = tf.reshape(up_score, (-1, 21))
    return up_score 
def fcn8(data_input):

    conv1_1 = tflearn.conv_2d(data_input, nb_filter = 64, filter_size = 3, strides = 1, regularizer = 'L2',activation = 'relu' )
    conv1_2 = tflearn.conv_2d(conv1_1, nb_filter = 64, filter_size = 3, strides = 1, regularizer = 'L2',activation = 'relu' )
    pool1 = tflearn.max_pool_2d(conv1_2, kernel_size = 3)
