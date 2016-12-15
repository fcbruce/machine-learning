#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Tue 13 Dec 2016 11:11:35
#
#

import tensorflow as tf
import numpy as np

from load_data import *

columns = 128 * 128

x = tf.placeholder(tf.float32, shape=[None, columns])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_vaiable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_kxk(x, k):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# input layer 128 x 128 x 1
x_image = tf.reshape(x, [-1, 128, 128, 1])

# hidden layer #1
# conv: kernel 7 x 7
# pool: max, kernel 3 x 3
# output: 43 x 43 x 32
W_conv1 = weight_variable([7, 7, 1, 32])
b_conv1 = bias_vaiable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_kxk(h_conv1, 3)

# hidden layer #2
# conv: kernel 5 x 5
# pool: max, kernel 3 x 3
# output: 15 x 15 x 64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_vaiable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_kxk(h_conv2, 3)

# hidden layer #3
# conv: kernel 3 x 3
# pool: max, kernel 3 x 3
# output: 5 x 5 x 256
W_conv3 = weight_variable([3, 3, 64, 256])
b_conv3 = bias_vaiable([256])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_kxk(h_conv3, 3)

# hidden layer #4
# full connection
# output: 1024
W_fc1 = weight_variable([5 * 5 * 256, 1024])
b_fc1 = bias_vaiable([1024])
h_pool3_flat = tf.reshape(h_pool3, [-1, 5 * 5 * 256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer
W_fc2 = weight_variable([1024, 1])
b_fc2 = bias_vaiable([1])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

pred = tf.greater(tf.sigmoid(y_conv), 0.5)
correct_pred = tf.equal(tf.cast(pred, tf.float32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


yy = 1 - tf.sigmoid(y_conv)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, "../model/adam1e-3/model_4000.ckpt")

"""
for i in range(4001, 5001):
    train_batch = get_train_batch(i)
    if i % 50 == 0:
        acc = accuracy.eval(feed_dict={
            x: train_batch[0], y_: train_batch[1], keep_prob: 1.0})
        print "step #%d: train accuracy %f" % (i, acc)
    if i % 100 == 0:
        test_batch = get_test_batch(i)
        acc = accuracy.eval(feed_dict={
            x: test_batch[0], y_: test_batch[1], keep_prob: 1.0})
        print "step #%d: test accuracy %f" % (i, acc)
    feed_dict={x: train_batch[0], y_: train_batch[1], keep_prob: 0.5}
    train_step.run(feed_dict=feed_dict)

    # loss = sess.run([cross_entropy], feed_dict=feed_dict)
    # print "step #%d: loss = %.2f" % (i, loss[0])
    if i % 1000 == 0:
        path = saver.save(sess, "../model/adam1e-3/model_%d.ckpt" % i)

#saver.restore(sess, "../model/adam1e-3/model.ckpt")

sum_acc = 0.
times = 0

for i in range(0, 10001, i):
    test_batch = get_test_batch(i)
    acc = accuracy.eval(feed_dict={
        x: test_batch[0], y_: test_batch[1], keep_prob: 1.0})
    sum_acc += acc
    times += 1

print "final accuarcy %f" % (sum_acc / times)
"""

test_batch = np.load('../data/test_data.npy')
final_result = []
for i in range(125):
    batch = test_batch[i * 100 : i * 100 + 100, :]
    result = yy.eval(feed_dict={x: batch, keep_prob: 1.0})
    final_result.append(result)

final_result = np.append([], final_result)

print 'id,label'

for i in range(12500):
    print "%d,%f" % (i + 1, final_result[i])
    

sess.close()
