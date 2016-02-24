#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Wed 24 Feb 2016 08:20:47 PM CST
#
#

import numpy as np

def find_closest_centroids(X, centroids):

    m, n = X.shape
    idx = [0 for i in range(m)]

    for i in range(m):
        x = X[i, :]
        dis2 = np.add.reduce(np.square(x - centroids), axis = 1)
        idx[i] = np.argmin(dis2)

    return idx

def compute_centroids(X, idx, k):
    
    m, n = X.shape

    centroids = np.zeros((k, n))
    count = np.zeros((k, 1))

    for i in range(m):
        centroids[idx[i], :] += X[i, :]
        count[idx[i]] += 1

    centroids /= count

    return centroids

def init_centroids(X, k):

    m, n = X.shape

    centroids = X.copy()

    np.random.shuffle(centroids)

    return centroids[:k, :]

if __name__ == '__main__':
    x = np.array([[1,1],[2,2],[3,3],[7,7],[8,8],[9,9]])
    centroids = init_centroids(x, 2)
    print centroids
    
    idx = find_closest_centroids(x, centroids)
    
    print idx
    centroids=compute_centroids(x, idx, 2)
    print centroids

