import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tflearn.data_utils import *
import config
FLAGS = tf.app.flags.FLAGS
VOC_DIR = FLAGS.voc_dir
def create_image_list(list_path):
    '''
    Write the voc image paths to txt file
    '''
    w = open(os.path.join(list_path, 'images.txt'), 'w')
    images_list = os.listdir(os.path.join(VOC_DIR, 'SegmentationClass'))
    for img in images_list:
        w.write(img+'\n')
def get_images_list():
    '''
    get the voc semantic image list.
    '''
    images_list = os.listdir(os.path.join(VOC_DIR, 'SegmentationClass'))
    return images_list
def transform_label(labels):
    '''
    Transform the label 255 in voc images for segmentation to 0.
    '''
    ans = np.zeros(labels.shape)
    for i in range(labels.shape[0]):
        if labels[i] == 255:
            ans[i] = 0
        else:
            ans[i] = labels[i]
    return ans
def load_data( batch_idx, batch_size):
    '''
    Load data and label from image paths.
    '''
    image_paths = get_images_list()
    num_samples = len(image_paths)
    j = batch_idx*batch_size%num_samples 
    images = []
    labels = []
    if j+batch_size <= num_samples:
        for idx in range(j,j+batch_size):
            print image_paths[idx]
            im = Image.open(os.path.join(VOC_DIR, 'JPEGImages', image_paths[idx].split('.')[0]+'.jpg'))
            label = Image.open(os.path.join(VOC_DIR, 'SegmentationClass', image_paths[idx]))
            im = np.array(im)
            label = np.array(label)
            images.append(im)
            labels.append(label)
    else:
        for idx in range(j,num_samples):
            im = Image.open(os.path.join(VOC_DIR, 'JPEGImages', image_paths[idx].split('.')[0]+'jpg'))
            label = Image.open(os.path.join(VOC_DIR, 'SegmentationClass', image_paths[idx]))
            im = np.array(im)
            images.append(im)
            labels.append(label)
        for idx in range(num_samples -j):
            im = Image.open(os.path.join(VOC_DIR, 'JPEGImages', images_paths[idx]))
            label = Image.open(os.path.join(VOC_DIR, 'SegmentationClass', image_paths[idx]))
            im = np.array(im)
            images.append(im)
            labels.append(label)
    labels = np.array(labels)
    labels = labels.ravel()
    labels = transform_label(labels)
    return np.array(images), np.array(labels)
if __name__ == '__main__':
    s,t = load_data(1, 20)
    print s[18].shape
    print t[18].shape
