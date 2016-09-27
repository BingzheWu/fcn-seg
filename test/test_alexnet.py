from models import alexnet_fcn_voc
import numpy as np
import tensorflow as tf
from data_utils.voc_utils import load_data
if __name__ == '__main__':
    labels, data = load_data(2000,1)
    #input_data = tf.placeholder("float",list(data.shape))
    shape =  data.shape
    input_data = tf.placeholder("float",[1,shape[1], shape[2],3])
    #data = np.ones([1,227,227,3])
    score = alexnet_fcn_voc(input_data)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        x = sess.run(score ,feed_dict = {input_data:data})
        print x.shape 
