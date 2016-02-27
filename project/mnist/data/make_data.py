#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Sat 27 Feb 2016 07:40:40 AM CST
#
#

import struct as st
import numpy as np

files = ['train-labels-idx1-ubyte', 'train-images-idx3-ubyte', 't10k-labels-idx1-ubyte', 't10k-images-idx3-ubyte']

for name in files:
    with open(name, 'rb') as f:
        data = f.read(8)
        magic, n = st.unpack('>ii', data)

        if magic == 2049: # labels
            data = f.read(n)
            labels = np.array(st.unpack('>' + 'B' * n, data), dtype=np.uint8)
            y = np.zeros((n, 10), dtype=np.int8)
            y[np.arange(n), labels] = 1

            np.save(name + '.npy', y)

        if magic == 2051: # images
            data = f.read(8)
            h, w = st.unpack('>ii', data)
            size = n * h * w
            data = f.read(size)
            images = np.array(st.unpack('>' + 'B' * size, data), dtype=np.uint8).reshape(n, -1)

            np.save(name + '.npy', images)

