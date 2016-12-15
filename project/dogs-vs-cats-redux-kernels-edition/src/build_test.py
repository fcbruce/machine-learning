#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Thu 15 Dec 2016 18:07:36
#
#

import numpy as np

import skimage.io as skio
import skimage.color as skcr
import skimage.transform as sktf

size = 128
shape = (size, size)

all_img = skio.imread_collection('../data/test/*.jpg')

print 'all test image load'

def rgb2grey_reshape(img):
    img_resize = sktf.resize(img, shape)
    return skcr.rgb2grey(img_resize)

imgs = [rgb2grey_reshape(img).reshape(-1) for img in all_img]

data = np.array(imgs, dtype=np.float32)

print data.shape

np.save('../data/test_data.npy', data)
