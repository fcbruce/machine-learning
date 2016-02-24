#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Wed 24 Feb 2016 10:53:31 PM CST
#
#

import skimage.io as skio
import skimage.data as skdata
import kmeans

lena = skdata.lena()

skio.imsave('lena.png', lena)

X = lena.reshape(-1, 3).copy()
m, n = X.shape

k = 8

centroids = kmeans.init_centroids(X, k)

iter = 0
while True:
    iter += 1
    print 'iteration ', iter
    idx = kmeans.find_closest_centroids(X, centroids)
    new_centroids = kmeans.compute_centroids(X, idx, k)

    if (new_centroids == centroids).all(): break

    centroids = new_centroids

for i in range(m):
    X[i, :] = centroids[idx[i], :]

X = X.reshape(lena.shape)
skio.imshow(X)
skio.show()
skio.imsave('compress_lena.png', X)
