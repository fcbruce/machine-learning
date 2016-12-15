#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Wed 14 Dec 2016 11:09:11
#
#

import numpy as np

data_file = '../data/data_128_%d.npy'
label_file = '../data/label_128_%d.npy'

def get_train_batch(i):
    n = i % 100
    m = i / 100 % 10
    data = np.load(data_file % n)[m * 100: m * 100 + 100, :]
    label = np.load(label_file % n)[m * 100: m * 100 + 100]
    return data, label.reshape((-1, 1))
    
def get_test_batch(i):
    n = i / 100 % 100
    data = np.load(data_file % n)[1000:1200, :]
    label = np.load(label_file % n)[1000:1200]
    return data, label.reshape((-1, 1))

def get_all_test():
    data = np.zeros((20000, 128 * 128), dtype=np.float32)
    label = np.zeros((20000), dtype=np.float32)

    for i in range(100):
        data[i * 100: i * 100 + 200, ] = np.load(data_file % i)[1000: 1200, :]
        label[i * 100: i * 100 + 200] = np.load(label_file % i)[1000: 1200]

    return data, label.reshape((-1, 1))

