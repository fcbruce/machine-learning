#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Mon 12 Dec 2016 20:52:30
#
#

import numpy as np
import random as rd

data_file_prefix = '../data/data_128_%d.npy'
label_file_prefix = '../data/label_128_%d.npy'

gap_list = [3, 7, 11, 13, 17, 19, 23, 29, 31, 37, 39, 41, 43, 47, 53, 57, 59]

rd.shuffle(gap_list)

for gap in gap_list:
    for i in range(100):
        j = (i + gap) % 100
        data1 = data_file_prefix % i
        label1 = label_file_prefix % i

        data2 = data_file_prefix % j
        label2 = label_file_prefix % j

        d1 = np.load(data1)
        l1 = np.load(label1)

        d2 = np.load(data2)
        l2 = np.load(label2)
        
        begin = rd.randint(0, 1000)
        end = min(begin + 233, 1200)

        d1[begin:end], d2[begin:end] = d2[begin:end], d1[begin:end]
        l1[begin:end], l2[begin:end] = l2[begin:end], l1[begin:end]

        np.save(data1, d1)
        np.save(data2, d2)
        np.save(label1, l1)
        np.save(label2, l2)

