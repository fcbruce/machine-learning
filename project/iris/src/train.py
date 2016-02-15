#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Mon 15 Feb 2016 12:24:23 PM CST
#
#

from gradient_descend import gradient_descend
import numpy as np

def train(X, y, alpha, lam, iter_num):
    k_, k = y.shape
    m, n = X.shape
    all_theta = np.matrix(np.zeros((k, n)))

    for i in range(iter_num):
        for j in range(k):
            theta = all_theta[j, :]
            theta = gradient_descend(theta, X, y[:, j], alpha, lam)
            all_theta[j, :] = theta

    return all_theta
