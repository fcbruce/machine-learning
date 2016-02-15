#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Mon 15 Feb 2016 03:51:06 PM CST
#
#

import util
from train import train
from predict import predict
import numpy as np

X_train = np.matrix(np.load('../data/X_train.npy'))
y_train = np.matrix(np.load('../data/y_train.npy'))
X_cv = np.matrix(np.load('../data/X_cv.npy'))
y_cv = np.matrix(np.load('../data/y_cv.npy'))
X_test = np.matrix(np.load('../data/X_test.npy'))
y_test = np.matrix(np.load('../data/y_test.npy'))

alpha = 0.1
lam = 0.1
iter_num = 100

all_theta = train(X_train, y_train, alpha, lam, iter_num)

#p = predict(all_theta, X_cv)
#accurancy = util.accurancy(p, y_cv)
#print "accurancy of cv: %.2f%%" % (accurancy * 100)

p = predict(all_theta, X_test)
accurancy = util.accurancy(p, y_test)
print "accurancy of test: %.2f%%" % (accurancy * 100)
