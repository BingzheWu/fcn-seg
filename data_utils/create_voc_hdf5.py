'''
This is a script for creating hdf5 files of VOC segmentation train and
val data set.
author: Bingzhe Wu
'''
import h5py
import copy
import os
import numpy as np
from PIL import Image
from voc_helper import voc
voc_dir = '~/bingzhe/data/VOC2012/'#voc_dir is /path/to/VOC2012
voc_ = voc(voc_dir)

def create_hdf5(mode, save_path):
    '''
    if mode is 'image', this code will generate the image data.
    Otherwise, the label data will be generated.
    '''
    images = []
    image_list = os.listdir(voc_.dir)
    for image in image_list:
        if mode == 'image':
            tmp = voc_.load_image(image.split('jpg')[0])
        else:
            tmp = voc_.load_label(image.split('jpg')[0])
        images.append(tmp)
    images = np.array(images)
    h5f = h5py.File(save_path+'voc_'+mode, 'w')
    h5f.create_dataset('voc_'+mode, data = images) 
if __name__ == '__main__':
    save_path = '~/bingzhe/data/voc_hdf5/'
    mode = 'image'
    create_hdf5(mode, save_path)




