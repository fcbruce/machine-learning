#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Mon 15 Feb 2016 05:54:43 PM CST
#
#

def accurancy(prediction, ground_truth):
    m, k = prediction.shape

    accurancy = 0.0

    for i in range(m):
        if (prediction[i, :] == ground_truth[i, :]).all():
            accurancy += 1

    accurancy /= m

    return accurancy
