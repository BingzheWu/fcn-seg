import numpy
import h5py

if __name__ == '__main__':
    hdf5_file = '/home/ceca/bingzhe/data/voc_hdf5/voc_image'
    hf5 = h5py.File(hdf5_file, 'r')
    tmp = hf5['voc_image']
    print tmp[1].shape
