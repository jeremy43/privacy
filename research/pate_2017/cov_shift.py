
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
import pickle
from scipy import linalg as LA
from sklearn.preprocessing import normalize
import cov_shift
def predict_data(dataset, nb_teachers, teacher = False):
  """
  This is for obtaining the weight from student / teache, don't involve any noise
  :param dataset:  string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :param teacher: if teacher is true, then predict with training dataset, else students
  :return: out prediction based on cnn
  """
  assert input.create_dir_if_needed(FLAGS.train_dir)
  if teacher:
    train_only = True
    test_only  = False
  else:
    train_only = False
    test_only =True

  # Load the dataset
  if dataset == 'svhn':
    test_data, test_labels = input.ld_svhn(test_only, train_only)
  elif dataset == 'cifar10':
    test_data, test_labels = input.ld_cifar10(test_only, train_only)
  elif dataset == 'mnist':
    test_data, test_labels = input.ld_mnist(test_only, train_only)
  else:
    print("Check value of dataset flag")
    return False

  teachers_preds = ensemble_preds(dataset, nb_teachers, test_data)

  # Aggregate teacher predictions to get student training labels
  pred_labels = aggregation.noisy_max(teachers_preds, 0)
  # Print accuracy of aggregated labels
  ac_ag_labels = metrics.accuracy(pred_labels, test_labels)
  print("obtain_weight Accuracy of the aggregated labels: " + str(ac_ag_labels))
  return test_data, pred_labels, test_labels

def prepare_student_data(dataset, nb_teachers, save=False, shift_data =None):
  """
  Takes a dataset name and the size of the teacher ensemble and prepares
  training data for the student model, according to parameters indicated
  in flags above.
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :param save: if set to True, will dump student training labels predicted by
               the ensemble of teachers (with Laplacian noise) as npy files.
               It also dumps the clean votes for each class (without noise) and
               the labels assigned by teachers
  :return: pairs of (data, labels) to be used for student training and testing
  """
  assert input.create_dir_if_needed(FLAGS.train_dir)

  # Load the dataset
  if dataset == 'svhn':
    test_data, test_labels = input.ld_svhn(test_only=True)
  elif dataset == 'cifar10':
    test_data, test_labels = input.ld_cifar10(test_only=True)
  elif dataset == 'mnist':
    test_data, test_labels = input.ld_mnist(test_only=True)
  else:
    print("Check value of dataset flag")
    return False

  # Make sure there is data leftover to be used as a test set
  assert FLAGS.stdnt_share < len(test_data)

  # Prepare [unlabeled] student training data (subset of test set)
  stdnt_data = test_data

  if shift_data is not None:
      #no noise
    # replace original student data with shift data

    stdnt_data = shift_data['data']
    test_labels = shift_data['label']
    print('*** length of shift_data {} lable length={}********'.format(len(stdnt_data),len(test_labels)))

  # Compute teacher predictions for student training data
  teachers_preds = ensemble_preds(dataset, nb_teachers, stdnt_data)

  # Aggregate teacher predictions to get student training labels
  if not save:
    stdnt_labels = aggregation.noisy_max(teachers_preds, FLAGS.lap_scale)
  else:
    # Request clean votes and clean labels as well
    stdnt_labels, clean_votes, labels_for_dump = aggregation.noisy_max(teachers_preds, FLAGS.lap_scale, return_clean_votes=True) #NOLINT(long-line)

    # Prepare filepath for numpy dump of clean votes
    filepath = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_student_clean_votes_lap_' + str(FLAGS.lap_scale) + '.npy'  # NOLINT(long-line)

    # Prepare filepath for numpy dump of clean labels
    filepath_labels = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_teachers_labels_lap_' + str(FLAGS.lap_scale) + '.npy'  # NOLINT(long-line)

    # Dump clean_votes array
    with tf.gfile.Open(filepath, mode='w') as file_obj:
      np.save(file_obj, clean_votes)

    # Dump labels_for_dump array
    with tf.gfile.Open(filepath_labels, mode='w') as file_obj:
      np.save(file_obj, labels_for_dump)

  # Print accuracy of aggregated labels
  if shift_data is not None:
    ac_ag_labels = metrics.accuracy(stdnt_labels, test_labels)
    print("Accuracy of the aggregated labels: " + str(ac_ag_labels))



  if save:
    # Prepare filepath for numpy dump of labels produced by noisy aggregation
    filepath = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_student_labels_lap_' + str(FLAGS.lap_scale) + '.npy' #NOLINT(long-line)

    # Dump student noisy labels array
    with tf.gfile.Open(filepath, mode='w') as file_obj:
      np.save(file_obj, stdnt_labels)

  return stdnt_data, stdnt_labels


def pca_transform(dataset, FLAGS):
    """
    Do PCA transform on both teacher and student dataset
    :param dataset:
    :return:
    pca transformed teacher and student dataset
    """
    teacher_file_name = FLAGS.data + 'PCA_teacher' + dataset + '.pkl'
    student_file_name = FLAGS.data + 'PCA_student' + dataset + '.pkl'
    if os.path.exists(teacher_file_name):
        return
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
    pca = PCA(n_components=dim)
    pca.fit(test_data)
    test_data = pca.transform(test_data)
    train_data = pca.transform(train_data)
    """
    normalize_matrix = normalize(test_data, axis=1, norm='l2')
    cov_matrix = np.matmul(np.transpose(normalize_matrix), normalize_matrix)
    evals, evecs = LA.eigh(cov_matrix)
    idx = np.argsort(evals)[::-1]
    #evecs = evecs[:, idx[:60]]
    test_data = np.matmul(test_data, evecs)
    train_data = normalize(train_data.reshape((-1, 784)), axis=1, norm='l2')
    train_data = np.matmul(train_data, evecs)
    """
    train_data = np.reshape(train_data, ori_train)
    test_data = np.reshape(test_data, ori_test)
    f = open(teacher_file_name,'wb')
    pickle.dump(train_data, f)
    f = open(student_file_name, 'wb')
    pickle.dump(test_data, f)


def prepare_cov_shift(dataset, a,b):
    """
    This function use pca to shift students' dataset, dataset size is the same
    first do pca on dataset, and choose the maximum projection
    compute the min value on min_v, and mean_v on that direction.
    apply a normal distribution with mu= min_v + (mean_v - min_v)/a, var = (mean_v - min_v)/b
    :param student_data:
    :return: student data with shift y
    """

    test_only = True
    train_only = False
    # Load the dataset
    if dataset == 'svhn':
        stdnt_data, test_labels = input.ld_svhn(test_only, train_only)
    elif dataset == 'cifar10':
        stdnt_data, test_labels = input.ld_cifar10(test_only, train_only)
    elif dataset == 'mnist':
        stdnt_data, test_labels = input.ld_mnist(test_only, train_only)
    else:
        print("Check value of dataset flag")
        return False
    from sklearn.decomposition import PCA, KernelPCA
    stdnt_data = stdnt_data[:1000,:]
    test_labels = test_labels[:1000]
    original_shape = stdnt_data.shape
    stdnt_data = np.reshape(stdnt_data, (-1,784)) #only for mnist
    dim = stdnt_data.shape[1]
    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
    projection = kpca.fit_transform(stdnt_data)
    #pca = PCA(n_components=dim)
    #pca.fit(stdnt_data)
    #projection = pca.transform(stdnt_data)
    # pick the maximum column
    max_col = projection[:,0]
    min_v = min(max_col)
    mean_v = np.mean(max_col)
    mu = min_v + (mean_v - min_v)/a
    var = (mean_v -min_v)/b
    new_col = np.random.normal(mu,var, stdnt_data.shape[0])
    projection[:,0] = new_col
    X_back = kpca.inverse_transform(projection)
    X_back = np.reshape(X_back,original_shape)
    shift_dataset = {}
    shift_dataset['data'] = X_back
    shift_dataset['label'] = test_labels
    return shift_dataset

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
def train_student(dataset, nb_teachers,inverse_w = None, shift_dataset = None):
  """
  This function trains a student using predictions made by an ensemble of
  teachers. The student and teacher models are trained using the same
  neural network architecture.
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :return: True if student training went well
  """
  assert input.create_dir_if_needed(FLAGS.train_dir)
  print('len of shift data'.format(len(shift_dataset['data'])))
  # Call helper function to prepare student data using teacher predictions
  stdnt_data, stdnt_labels= prepare_student_data(dataset, nb_teachers, save=True, shift_data = shift_dataset)

  # Unpack the student dataset, here stdnt_labels are already the ensemble noisy version
  # Prepare checkpoint filename and path
  if FLAGS.deeper:
    ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student_deeper.ckpt' #NOLINT(long-line)
  else:
    ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student.ckpt'  # NOLINT(long-line)

  # Start student training
  weights = np.zeros(len(stdnt_data))
  print('len of weight={} len of labels= {} '.format(len(weights), len(stdnt_labels)))
  for i, x in enumerate(weights):
    weights[i] = np.float32(inverse_w[stdnt_labels[i]])
 # assert deep_cnn.train(stdnt_data, stdnt_labels, ckpt_path, weights= weights)
  deep_cnn.train(stdnt_data, stdnt_labels, ckpt_path)
  # Compute final checkpoint name for student (with max number of steps)
  ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)
  private_data, private_labels = input.ld_mnist(test_only = False, train_only = True)
  # Compute student label predictions on remaining chunk of test set
  teacher_preds = deep_cnn.softmax_preds(private_data, ckpt_path_final)
  student_preds =  deep_cnn.softmax_preds(stdnt_data, ckpt_path_final)
  # Compute teacher accuracy
  precision_t = metrics.accuracy(teacher_preds, private_labels)
  precision_s  = metrics.accuracy(student_preds, stdnt_labels)
  print('shift_ratio={} Precision of teacher after training:{} student={}'.format(shift_dataset['shift_ratio'], precision_t, precision_s))

  return True

