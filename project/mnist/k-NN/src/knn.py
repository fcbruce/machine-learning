#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Tue 08 Mar 2016 21:51:43
#
#

import numpy as np

def classify(X, samples, labels, k):
    
    diff = (X - samples) ** 2
    sqDis = diff.sum(axis=1)
    
    idx = sqDis.argsort();

    classCount = {};
    for i in range(k):
        classCount[labels[idx[i]]] = classCount.get(labels[idx[i]], 0) + 1

    return max(classCount, key=lambda k:classCount[k])



