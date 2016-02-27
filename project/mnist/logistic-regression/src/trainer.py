#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Sat 27 Feb 2016 05:30:23 PM CST
#
#

from gradient_descent import *
from util import predict
from sklearn.metrics import accuracy_score, precision_score, recall_score
import random
import json
from datetime import datetime

class Trainer:

    def __init__(self, file_path):

        f = open(file_path, 'r')
        data = json.load(f)

        self.batch = data['batch size']
        self.iter_num = data['iter number']
        self.gap = data['test gap']
        self.alpha = data['learning rate']
        self.lam = data['weight decay']

        train_size = data['train size']
        images_train = np.load(data['images-train'])
        labels_train = np.load(data['labels-train'])
        self.X_train = images_train[:train_size]
        self.y_train = labels_train[:train_size]
        self.X_cv = images_train[train_size:]
        self.y_cv = labels_train[train_size:]

        self.X_test = np.load(data['images-test'])
        self.y_test = np.load(data['labels-test'])

        self.acc = []
        self.J = []

        m, n = self.X_train.shape
        m, k = self.y_train.shape

        if 'weight-file' not in data:
            self.all_theta = np.zeros((k, n + 1))
        else:
            self.all_theta = np.load(data['weight-file'])

        if 'save-module-file' not in data:
            self.save_file = '../../data/' + datetime.now().strftime('%Y%m%d_%H%M') + '.npy'
        else:
            self.save_file = data['save-module-file']


    def test(self, X, y):

        prediction = predict(self.all_theta, X)

        acc = accuracy_score(np.argmax(y,axis=1), prediction)
        # prec = precision(y, prediction)
        # recall = recall(y, prediction)

        print 'accuracy : %f%%' % (acc * 100)

        self.acc.append(acc)


    def train(self):

        m, n = self.X_train.shape
        m, k = self.y_train.shape

        X = np.hstack((np.ones((m, 1), dtype=np.int8), self.X_train))
        y = self.y_train
        
        m, n = X.shape
        for i in range(self.iter_num):

            if i % self.gap == 0:
                print 'Iteration #%d test:' % (i)
                self.test(self.X_cv, self.y_cv)

            for j in range(k):
                
                begin = random.randint(0, m - self.batch)

                J, grad = cost_function(self.all_theta[j, :], X[begin:begin + self.batch, :], y[begin:begin + self.batch, j], self.lam)
                self.J.append(J)

                self.all_theta[j, :] = gradient_descent(self.all_theta[j, :], grad, self.alpha)

        print 'Iteration #%d test:' % (self.iter_num)
        self.test(self.X_cv, self.y_cv)

        print 'Done!'

        np.save(self.save_file, self.all_theta)

        return self.all_theta
