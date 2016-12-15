#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Fri 25 Nov 2016 16:07:36
#
#

import os
import skimage.io as skio
import skimage.color as skcr
import skimage.transform as sktf
import numpy as np

import random as rd

data_npy = '../data/data_128_%d.npy'
label_npy = '../data/label_128_%d.npy'

cats_files = '../data/train/cat.*.jpg';
dogs_files = '../data/train/dog.*.jpg';

size = 128
shape = (size, size)

cats_all = skio.imread_collection(cats_files)
dogs_all = skio.imread_collection(dogs_files)

print 'loaded all data'

def rgb2grey_reshape(img): 
    img_resize = sktf.resize(img, shape)
    return skcr.rgb2grey(img_resize)

def rotate(img, angle):
    img_rotate = sktf.rotate(img, angle)
    return rgb2grey_reshape(img_rotate)

def rotate_flip(img, angle):
    return np.fliplr(rotate(img, angle))

def expend2list(rgb):
    img = rgb2grey_reshape(rgb)
    return [rotate(rgb, 10), rotate(rgb, -10), np.fliplr(img), img, rotate_flip(rgb, 10), rotate_flip(rgb, -10)]

for i in range(0, 10000, 100):

    print '\ncreate the image block #%d' % (i / 100)
    dogs = dogs_all[i : i + 100]
    grey_dogs_lists = [expend2list(dog) for dog in dogs]
    grey_dogs = [(dog.reshape(-1), 0) for sub_dogs in grey_dogs_lists for dog in sub_dogs]
    rd.shuffle(grey_dogs)

    print 'dogs ok'

    cats = cats_all[i : i + 100]
    grey_cats_lists = [expend2list(cat) for cat in cats]
    grey_cats = [(cat.reshape(-1), 1) for sub_cats in grey_cats_lists for cat in sub_cats]
    rd.shuffle(grey_cats)

    print 'cats ok'

    data_set = grey_dogs + grey_cats

    rd.shuffle(data_set)

    data, label = zip(*data_set)

    data = np.array(data)
    label = np.array(label)

    np.save(data_npy % (i / 100), data)
    np.save(label_npy % (i / 100), label)

    print 'the image block #%d ok' % (i / 100)

