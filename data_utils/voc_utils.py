import os
import numpy as np
from PIL import Image
VOC_DIR = '/home/ceca/bingzhe/data/VOC2012'
def create_image_list(list_path):
    w = open(os.path.join(list_path, 'images.txt'), 'w')
    images_list = os.listdir(os.path.join(VOC_DIR, 'SegmentationClass'))
    for img in images_list:
        w.write(img+'\n')
def get_images_list():
    images_list = os.listdir(os.path.join(VOC_DIR, 'SegmentationClass'))
    return images_list
def load_data( batch_idx, batch_size):
    image_paths = get_images_list()
    num_samples = len(image_paths)
    j = batch_idx*batch_size%num_samples 
    images = []
    labels = []
    if j+batch_size <= num_samples:
        for idx in range(j,j+batch_size):
            im = Image.open(os.path.join(VOC_DIR, 'SegmentationClass', image_paths[idx]))
            label = Image.open(os.path.join(VOC_DIR, 'JPEGImages', image_paths[idx].split('.')[0]+'.jpg'))
            im = np.array(im)
            label = np.array(label)
            images.append(im)
            labels.append(label)
    else:
        for idx in range(j,num_samples):
            im = Image.open(os.path.join(VOC_DIR, 'SegmentationClass', images_paths[idx]))
            label = Image.open(os.path.join(VOC_DIR, 'JPEGImages', image_paths[idx]))
            im = np.array(im)
            images.append(im)
            labels.append(label)
        for idx in range(num_samples -j):
            im = Image.open(os.path.join(VOC_DIR, 'SegmentationClass', images_paths[idx]))
            label = Image.open(os.path.join(VOC_DIR, 'JPEGImages', image_paths[idx]))
            im = np.array(im)
            images.append(im)
            labels.append(label)
    return np.array(images), np.array(labels)
if __name__ == '__main__':
    s,t = load_data(1, 20)
    print s[0].shape
    print t[0].shape
