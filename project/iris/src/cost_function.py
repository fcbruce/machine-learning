#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Mon 15 Feb 2016 11:10:20 AM CST
#
#

import numpy as np
from sigmoid import sigmoid

def cost_function(theta, X, y, lam):
    m, n = X.shape

    hx = sigmoid(X * theta.T)

    J = sum(-y.T * np.log(hx) - (1 - y.T) * np.log(1 - hx)) / m + lam * (theta.T * theta - theta[0, 0] ** 2) / (2 * m)

    grad = (hx - y).T * X / m + lam * theta / m
    grad[0, 0] -= lam * theta[0, 0] / m

    return J[0, 0], grad

