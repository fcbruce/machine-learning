#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Sat 27 Feb 2016 07:30:49 PM CST
#
#

import numpy as np
from sigmoid import sigmoid

def predict(all_theta, X):

    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    prediction = np.argmax(sigmoid(np.dot(X, all_theta.T)), axis=1)

    return prediction
