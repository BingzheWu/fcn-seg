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
voc_dir = '/home/ceca/bingzhe/data/VOC2012'#voc_dir is /path/to/VOC2012
voc_ = voc(voc_dir)

def create_hdf5(mode, save_path):
    '''
    if mode is 'image', this code will generate the image data.
    Otherwise, the label data will be generated.
    '''
    images = []
    h5f = h5py.File(save_path+'voc_'+mode, 'w')
    s = h5f.create_dataset('voc_'+mode,(1,500,500,3), maxshape = (None,None,None,None))
    if mode == 'image':
        image_list = os.listdir(os.path.join(voc_.dir,'JPEGImages'))
    else:
        image_list = os.listdir(os.path.join(voc_.dir, 'SegmentationClass'))
    idx = 0
    lenth = len(image_list)
    for image in image_list:
        print image
        if mode == 'image':
            tmp = voc_.load_image(image.split('.jpg')[0])
        else:
            tmp = voc_.load_label(image.split('.jpg')[0])
        tmp = tmp[np.newaxis,...]
        print tmp.shape
        s.resize((lenth,tmp.shape[1],tmp.shape[2],3))
        s[idx,...] = tmp
        idx = idx + 1
if __name__ == '__main__':
    save_path = '/home/ceca/bingzhe/data/voc_hdf5/'
    mode = 'image'
    create_hdf5(mode, save_path)




