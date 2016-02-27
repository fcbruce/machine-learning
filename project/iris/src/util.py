#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Mon 15 Feb 2016 05:54:43 PM CST
#
#

def accuracy(prediction, ground_truth):
    m, k = prediction.shape

    accuracy = 0.0

    for i in range(m):
        if (prediction[i, :] == ground_truth[i, :]).all():
            accuracy += 1

    accuracy /= m

    return accuracy
