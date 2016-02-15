#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Mon 15 Feb 2016 04:54:09 AM CST
#
#

import random
import numpy as np
import define

train_count = 90
cv_count = 30
test_count = 30

f = open('../data/iris.data', 'r')

data = []

for row in f:
    data.append(row[: -1].split(','))

random.shuffle(data)

data = np.array(data).reshape(-1, 5)

X = np.float64(data[:, : -1])
m, n = X.shape
X = np.hstack((np.ones((m, 1)), X))

yl = data[:, -1:]

y = np.zeros((m, 3), dtype = np.int32)


i = 0
for name in yl:
    y[i, define.TYPE[name[0]]] = 1
    i += 1

X_train = X[: train_count, :]
y_train = y[: train_count, :]
np.save('../data/X_train', X_train)
np.save('../data/y_train', y_train)

X_cv = X[train_count: train_count + cv_count, :]
y_cv = y[train_count: train_count + cv_count, :]
np.save('../data/X_cv', X_cv)
np.save('../data/y_cv', y_cv)

X_test = X[-test_count: , :]
y_test = y[-test_count: , :]
np.save('../data/X_test', X_test)
np.save('../data/y_test', y_test)


