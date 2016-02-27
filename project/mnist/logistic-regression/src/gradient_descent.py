#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Sat 27 Feb 2016 05:11:06 AM CST
#
#

import numpy as np
from sigmoid import sigmoid

# theta is a numpy-array of n + 1 by 1
# X is a numpy-array of m by n + 1
# y is a numpy-array of m by 1
# J is a value of cost
# grad is a numpy-array of n + 1 by 1
# only X is 2-d array
def cost_function(theta, X, y, lam):
    
    m, n = X.shape
    h = sigmoid(np.dot(X, theta))

    J = np.sum(-(np.dot(y, np.log(h))) - np.dot((1. - y), np.log(1. - h))) / m + lam * (sum(theta * theta) - theta[0] ** 2) / (2. * m)

    grad = np.dot(X.T, h - y) + lam * theta
    grad[0] -= theta[0] * lam
    grad /= m

    return J, grad

def gradient_descent(theta, grad, alpha):

    return theta - alpha * grad

if __name__ == '__main__':

    theta = np.array([1,1,1])
    X = np.array([[1, 5, 5]])
    y = np.array([1])

    J, grad = cost_function(theta, X, y, 1)

    print J
    print grad

    new_theta = gradient_descent(theta, grad, 0.1)
    print new_theta
