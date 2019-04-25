
import numpy as np
from six.moves import xrange
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import aggregation
import os
import deep_cnn
import input
from sklearn.decomposition import PCA, KernelPCA
import metrics
import scipy.stats
import pickle
from scipy import linalg as LA
from sklearn.preprocessing import normalize



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
    else:
        print("Check value of dataset flag")
        return False
    ori_train = train_data.shape
    ori_test = test_data.shape
    test_data = test_data.reshape((-1, dim))
    train_data = train_data.reshape((-1,dim))
    pca = PCA(n_components=1)
    pca.fit(test_data)
    max_component =pca.components_.T
    projection = np.dot(test_data, max_component)
    min_v = min(projection)
    mean_v = np.mean(projection)
    a = 1
    b = 1
    mu = min_v + (mean_v - min_v) / a
    var = (mean_v - min_v) / b
    prob = scipy.stats.norm(mu, var).pdf(projection)
    prob = np.ravel(prob.T) # transform into 1d dim
    index = np.where(prob>0)[0]
    sample = np.random.choice(index,len(index),replace = True,p = prob/sum(prob))
    test_data = test_data[sample]
    train_data = np.reshape(train_data, ori_train)
    test_data = np.reshape(test_data, ori_test)
    f = open(teacher_file_name,'wb')
    pickle.dump(train_data, f)
    f = open(student_file_name, 'wb')
    pickle.dump(test_data, f)
    print('finish pca transform')


def logistic(FLAGS):
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
    clf = LogisticRegression(penalty='l2', C=2, solver='sag', multi_class='ovr').fit(dataset, label)
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

