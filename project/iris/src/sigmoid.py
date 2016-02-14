#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Mon 15 Feb 2016 06:13:09 AM CST
#
#

import numpy as np

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

