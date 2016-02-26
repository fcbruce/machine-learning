#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Sat 27 Feb 2016 04:55:11 AM CST
#
#

import numpy as np

def sigmoid(z):

    return 1. / (1. + np.exp(-z))

if __name__ == '__main__':

    print 'sigmoid(10) = ' , sigmoid(10)
    print 'sigmoid(-10) = ' , sigmoid(-10)
