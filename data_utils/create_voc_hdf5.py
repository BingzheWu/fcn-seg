'''
This is a script for creating hdf5 files of VOC segmentation train and
val data set.
author: Bingzhe Wu
'''
import h5py
import cv2
import copy
import os
import numpy as np
from PIL import Image
from voc_helper import voc
voc_dir = '/home/ceca/bingzhe/data/VOC2012'#voc_dir is /path/to/VOC2012
voc_ = voc(voc_dir)
def load_image(idx):
    im = Image.open('{}/JPEGImages/{}.jpg'.format(voc_dir, idx))
    #im = cv2.imread('{}/JPEGImages/{}.jpg'.format(voc_dir, idx))
    #b,g,r = cv2.split(im)
    #im = cv2.merge([r,g,b])
    print im.size
    im = im.resize((224,224))
    return np.array(im)
def create_hdf5(mode, save_path):
    '''
    if mode is 'image', this code will generate the image data.
    Otherwise, the label data will be generated.
    '''
    h5f = h5py.File(save_path+'voc'+mode, 'w')
    if mode == 'image':
        image_list = os.listdir(os.path.join(voc_.dir,'JPEGImages'))
    else:
        image_list = os.listdir(os.path.join(voc_.dir, 'SegmentationClass'))
    idx = 0
    lenth = len(image_list)
    s = h5f.create_dataset('voc_'+mode,(lenth,224,224,3), maxshape = (lenth,224,224,3))
    print lenth
    for image in image_list:
        print image
        if mode == 'image':
            tmp = load_image(image.split('.jpg')[0])
        else:
            tmp = voc_.load_label(image.split('.')[0])
        #s.resize((lenth+2,tmp.shape[1],tmp.shape[2],3))
        print tmp.shape
        s[idx,...] = tmp
        idx = idx +1
        if idx > 1:
            break
    h5f.close()
if __name__ == '__main__':
    save_path = '/home/ceca/bingzhe/data/voc_hdf5/'
    mode = 'label'
    create_hdf5(mode, save_path)




