#
#
# Author : fcbruce <fcbruce8964@gmail.com>
#
# Time : Sat 27 Feb 2016 08:04:07 PM CST
#
#

from trainer import Trainer
from util import predict
from sklearn.metrics import accuracy_score
import numpy as np

trainer = Trainer('../module/module.json')
all_theta = trainer.train()

prediction = predict(all_theta, trainer.X_test)

acc = accuracy_score(np.argmax(trainer.y_test, axis=1), prediction)

print '\ntest...'
print 'accuracy : %f%%' % (acc * 100)
