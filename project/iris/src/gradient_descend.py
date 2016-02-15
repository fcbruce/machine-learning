#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Mon 15 Feb 2016 03:22:57 PM CST
#
#

from cost_function import cost_function

def gradient_descend(theta, X, y, alpha, lam):
    J, grad = cost_function(theta, X, y, lam)

    return theta - alpha * grad

