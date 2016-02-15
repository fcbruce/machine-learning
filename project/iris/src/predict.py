#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Mon 15 Feb 2016 03:39:19 PM CST
#
#

import numpy as np
from sigmoid import sigmoid

def predict(theta, X):
    h = sigmoid(X * theta.T)
    m, k = h.shape
    y = np.zeros(h.shape, dtype=np.int32)
    y = np.matrix(y)

    for i in range(m):
        y[i, np.argmax(h[i, :])] = 1

    return y

