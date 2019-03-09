import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
ctx = mx.cpu()

mnist = mx.test_utils.get_mnist()
train_data = mnist['train_data'].reshape([60000, -1])[:100, :]
train_label = mnist['train_label'][:100]
length = len(train_label)
test_data = mnist['test_data'].reshape([10000, -1])[:100, :]
test_label = mnist['test_label'][:100]

num_hidden = 256
num_inputs = len(train_data[0])
num_outputs = 10
Num_class = 10
"""
    Partition training dataset as three part
    X1, Y1 for f
    X2, Y2, and test set  for cov
    X3, Y3 for final test part
"""
X1= train_data[0:length,:]
Y1 = train_label[0:length]
X2 = train_data[length:2*length, :]
Y2 = train_label[length : 2*length]
X3 = train_data[2*length:,:]
Y3 = test_label[2*length :]

def train_f(X1, Y1):
    """

    :param X1: data
    :param Y1: label
    :return: net
    the net 256 hidden
    """
    net = gluon.nn.HybridSequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(num_hidden, in_units=num_inputs,activation="relu"))
        net.add(gluon.nn.Dense(num_outputs,in_units=num_hidden))

    # get and save the parameters
    params = net.collect_params()
    params.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    params.setattr('grad_req', 'write')
    epoches = 3
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    batch_size = 16
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
    for e in range(epoches):
        train_data = mx.io.NDArrayIter(X1, Y1,
                                       batch_size, shuffle=True)
        for i, batch in enumerate(train_data):
            data = batch.data[0].as_in_context(ctx).reshape((-1, 784))
            label = batch.label[0].as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)
    return net

def evaluate_f(data, label, net):
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    data_iterator = mx.io.NDArrayIter(X1, Y1, 64, shuffle=True)
    acc = mx.metric.Accuracy()
    loss_fun = .0
    data_iterator.reset()
    for i, batch in enumerate(data_iterator):
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
        loss = softmax_cross_entropy(output, label)
        loss_fun = loss_fun * i / (i + 1) + nd.mean(loss).asscalar() / (i + 1)
    return acc.get()[1], loss_fun

def obtain_weight(X2, Y2, test_data, net):
    """

    :param X2:
    :param Y2:
    :param test_data: test_data is public information without label
    :return: w which is the estimate q(y0/p(y)
    """
    test_data = mx.nd.array(test_data)
    output = net(test_data)
    estimate_mu = nd.argmax(output, axis = 1).asnumpy()
    np.reciprocal(estimate_mu, estimate_mu)
    X2 = mx.nd.array(X2)
    output = net(X2)
    f_y = nd.argmax(output, axis = 1)
    cov = np.zeros([Num_class, Num_class])
    for index, x in enumerate(f_y):
        cov[x, Y2[index]] += 1
    cov = cov / len(f_y)
    inverse_w = np.dot(cov, estimate_mu)
    return inverse_w

if __name__ == "__main__":
    net = train_f(X1, Y1)
    w = obtain_weight(X2, Y2, test_data, net)

