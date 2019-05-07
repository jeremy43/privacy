
import numpy as np
from six.moves import xrange
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import aggregation
from cvxpy import *
import os
import deep_cnn
import input
from sklearn.decomposition import PCA, KernelPCA
import metrics
import scipy.stats
import pickle
from scipy import linalg as LA
from sklearn.preprocessing import normalize


def evaluation(X, theta, y):
    """
    evaluate logistic regressian
    :param X:
    :param theta: class * dim
    :param y:
    :return:
    """
    num_cls = len(np.unique(y))
    new_col = np.ones([len(X), 1])  # for bias term, \belta* 1
    new_X = np.append(X, new_col, axis=1) #add bias term
    theta = np.array(theta)
    prob = np.dot(new_X, theta.transpose())
    name_y = np.unique(y)
    if num_cls == 2:
        indices = (prob > 0).astype(np.int)
    else:
        indices = prob.argmax(axis = 1)

    y_hat = name_y[indices]
    #y_hat = [x + np.min(y) for x in y_hat]
    sum = 0
    tf = 0
    tp = 0
    for idx in range(len(y)):
        if y[idx] == y_hat[idx]:
            sum+=1
            if y_hat[idx] == 1:
                tp +=1
            else:
                tf +=1
    accuracy = sum / len(y)
    print('accuracy = {} totoal = {} tf = {} tp ={}'.format(accuracy,len(y_hat),tf, tp))
    return accuracy

def cvx_objpert(X, y, eps, delta,weight = True):
    """

    :param X: Input X with n records
    :param Y: {1 or -1}
    :param eps:
    :param delta:
    :return: logistic parameter theta in k class
    prepocession dataset into num_cls classes (one vs all model), in each class y in (1, -1)

    to compute objpert for logistic regression
    L (lipschiz) = 1
    lamda = 1/4 (strong convex term)
    Lamda = 2*lamda/eps
    """
    theta = []
    num_cls = len(np.unique(y))
    theta = []
    new_col = np.ones([len(X), 1])
    X = np.append(X, new_col, axis=1)  # add bias term
    dim = X.shape[1]
    n = X.shape[0]
    L = 1
    lamda = 1/4
    Lamda = 2*lamda/eps
    b = np.sqrt(8*np.log(2/delta) + 4*eps)*L/eps * np.random.standard_normal(dim)

    for idx, cls in enumerate(np.unique(y)):
        if len(np.unique(y))==2:
            cls = np.unique(y)[1] # the latter one considered groundtruth
        idx = np.where(y == cls)
        nidx =np.where(y!= cls)
        cur_y = -np.ones(shape = y.shape)
        cur_y[idx] = 1
        wei_v = np.ones(shape=y.shape) #importance weight
        if weight == True:
            num_pos = len(idx[0])
            imp_weight = num_pos/(n - num_pos) # number of private set divide by students set
            print('num_prv : num_pub ={}'.format(imp_weight))
            wei_v = imp_weight*np.ones(shape = y.shape)
            wei_v[idx] = 1
        w = Variable(shape = dim)
        wei_v = multiply(cur_y, wei_v)
        print('cur_y={}'.format(cur_y))
        loss = sum(logistic(- X[idx]*w)) + sum(logistic(X[nidx]*w))+ Lamda/(2)*norm(w,2) +sum(multiply(b,w))
        #loss = sum(multiply(wei_v,logistic(-multiply(cur_y, X*w)))) + Lamda/(2)*norm(w,2) +sum(multiply(b,w))
        constraints =[norm(w,2) <=1]
        problem = Problem(Minimize(loss), constraints)
        try:
            problem.solve(verbose = True)
        except SolverError:
            problem.solve(verbose = True, solver = SCS)
        opt = problem.value
      #  print(opt)
        cur_theta = w.value
        #cur_theta = np.expand_dims(cur_theta, axis =0) #add dim for evaluate
        theta.append(cur_theta)
        if len(np.unique(y)) == 2:
            break
    theta = np.array(theta)
    return theta

def pca_transform(dataset, FLAGS):
    """
    Do PCA transform on both teacher and student dataset
    :param dataset:
    :return:
    pca transformed teacher and student dataset
    """
    teacher_file_name = FLAGS.data + '/PCA_teacher' + dataset + '.pkl'
    student_file_name = FLAGS.data + '/PCA_student' + dataset + '.pkl'
    #if os.path.exists(teacher_file_name):
        #return
    test_only = False
    train_only = False
    dim = 784
    # Load the dataset
    if dataset == 'svhn':
        train_data, train_labels, test_data, test_labels = input.ld_svhn(test_only, train_only)

    elif dataset == 'cifar10':
        train_data, train_labels, test_data, test_labels= input.ld_cifar10(test_only, train_only)
    elif dataset == 'mnist':
        train_data, train_labels, test_data, test_labels = input.ld_mnist(test_only, train_only)
        dim = 784
        test_data = test_data.reshape((-1, dim))
        train_data = train_data.reshape((-1, dim))
    elif dataset == 'adult':
        train_data, train_labels, test_data, test_labels = input.ld_adult(test_only, train_only)
        dim = 108
    else:
        print("Check value of dataset flag")
        return False
    ori_train = train_data.shape
    ori_test = test_data.shape
    pca = PCA(n_components=1)
    pca.fit(test_data)
    max_component =pca.components_.T
    projection = np.dot(test_data, max_component)
    min_v = np.min(projection)
    mean_v = np.mean(projection)
    a = 1e3
    b = 10
    mu = min_v + (mean_v - min_v) / a
    var = (mean_v - min_v) / b
    prob = scipy.stats.norm(mu, var).pdf(projection)
    prob = np.ravel(prob.T) # transform into 1d dim
    index = np.where(prob>0)[0]
    sample = np.random.choice(index,len(index),replace = True,p = prob/np.sum(prob))
    test_data = test_data[sample]
    train_data = np.reshape(train_data, ori_train)
    test_data = np.reshape(test_data, ori_test)
    f = open(teacher_file_name,'wb')
    pickle.dump(train_data, f)
    f = open(student_file_name, 'wb')
    pickle.dump(test_data, f)
    print('finish pca transform')


def cov_logistic(FLAGS):
    """
    use logistic regression to learn cov shift between teacher and student
     the label for teacher is 1, for student is -1
     p(z=1|x) = \frac{1}{1+e^{-f(x)}}
    :param teacher:
    :param student:
    :return:
    """

    teacher_file_name = FLAGS.data + 'PCA_teacher' + FLAGS.dataset + '.pkl'
    student_file_name = FLAGS.data + 'PCA_student' + FLAGS.dataset + '.pkl'
    f = open(teacher_file_name, 'rb')
    teacher = pickle.load(f)
    f = open(student_file_name, 'rb')
    student = pickle.load(f)
    assert input.create_dir_if_needed(FLAGS.train_dir)
    if FLAGS.dataset == 'mnist':
        student = student.reshape(-1, 784)
        teacher = teacher.reshape(-1, 784)

    y_t = np.ones(teacher.shape[0])
    y_t = np.expand_dims(y_t, axis=1)
    y_s = -np.ones(student.shape[0])
    y_s = np.expand_dims(y_s, axis=1)
    teacher = np.append(teacher, y_t, axis = 1)
    student = np.append(student, y_s, axis = 1)
    dataset = np.concatenate((teacher, student), axis = 0)
    np.random.shuffle(dataset)
    label = dataset[:,-1]
    dataset = dataset[:,:-1]

    coeff = cvx_objpert(dataset, label, FLAGS.eps_shift, FLAGS.delta_shift)
    ac = evaluation(dataset, coeff, label)
    print('accuracy of objpert={} eps ={} delta={}'.format(ac, FLAGS.eps_shift, FLAGS.delta_shift))
    clf = LogisticRegression(penalty='l2', C=2, solver='sag', multi_class='ovr').fit(dataset, label)
    print('non private  predict score for covshift = {}'.format(clf.score(dataset, label)))
    # add bias column for coef
    coeff = clf.coef_  # doesn't involve bias here, bias is self.intercept_
    bias = clf.intercept_
    bias = np.expand_dims(bias, axis=1)
    # coeff refer to theta star in paper, should be cls * d+1
    coeff = np.concatenate((coeff, bias), axis=1)
    coeff = np.squeeze(coeff)
    # importance weight = p(x)/q(x) = np.exp(f(x))
    weight = np.exp(np.dot(student,coeff.T))
    return weight

