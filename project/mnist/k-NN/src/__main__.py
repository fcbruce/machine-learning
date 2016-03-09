#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Tue 08 Mar 2016 22:21:27
#
#

import json
from knn import *
import numpy as np

if __name__ == '__main__':
    
    f = open('../module/knn.json', 'r')
    data = json.load(f)

    train_images = np.load(data['train-images']) / 255.
    train_labels = np.load(data['train-labels'])
    train_labels = train_labels.argmax(axis=1)

    test_images = np.load(data['test-images']) / 255.
    test_labels = np.load(data['test-labels'])
    test_labels = test_labels.argmax(axis=1)

    error = 0;

    for i in range(test_images.shape[0]):
    #for i in range(10):
        result = classify(test_images[i, :], train_images, train_labels, 100)

        if (result != test_labels[i]):
            error += 1.

    print error
