import mxnet as mx
import numpy as np
from mxnet import nd, autograd
from mxnet import gluon
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
f_size = 30
mnist = mx.test_utils.get_mnist()
train_data = mnist['train_data'].reshape([60000,-1])[:100,:]
train_label = mnist['train_label'][:100]
test_data = mnist['test_data'].reshape([10000,-1])[:100,:]
test_label = mnist['test_label'][:100]
Num_class  = 10
train_f = train_data[0:f_size,:]
label_f = train_label[0:f_size]
clf = LogisticRegression(penalty='l2', solver='sag', multi_class='multinomial').fit(train_f, label_f)
train_w = train_data[f_size:,: ]
label_w = train_label[f_size:]
# knock out some label = 5
index_out = np.where(test_label ==5)
ratio = 0.7
index_remain = index_out[0][0:int(ratio*len(index_out[0]))]
index_keep = np.where(test_label != 5)
index_keep = index_keep[0]
index = np.concatenate((index_remain, index_keep))
shift_data  = test_data[index]
shift_label = test_label[index]
tilde_y = clf.predict(train_w)
cov = np.zeros([Num_class, Num_class])
for index, x in enumerate(tilde_y):
    cov[x, label_w[index]] += 1
cov = cov/len(tilde_y)
np.linalg.cond(cov, p=-2)
predict_shift = clf.predict(shift_data)
tilde_mu = np.array([np.sum(predict_shift==x) for x in range(Num_class)])/len(predict_shift) + 1e-8 # 1e-8 is noise
np.reciprocal(tilde_mu,tilde_mu)
estimate_w = cov.dot(tilde_mu)
#s = np.random.dirichlet(0.1,20)

