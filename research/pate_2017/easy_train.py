# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
import os
import pickle
import utils
import aggregation
import deep_cnn
import input
import metrics
import cov_shift
FLAGS = tf.flags.FLAGS


tf.flags.DEFINE_string('dataset', 'svhn', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')
tf.flags.DEFINE_boolean('PATE2',True,'whether implement pate2')
tf.flags.DEFINE_string('data_dir','../data','Temporary storage')
tf.flags.DEFINE_string('train_dir','../model','Where model chkpt are saved')
tf.flags.DEFINE_string('teachers_dir','../model',
                       'Directory where teachers checkpoints are stored.')
tf.flags.DEFINE_string('data','../data/', 'where pca data are saved ')
tf.flags.DEFINE_integer('teachers_max_steps', 2500,
                        'Number of steps teachers were ran.')
tf.flags.DEFINE_integer('max_steps', 3500, 'Number of steps to run student.')
tf.flags.DEFINE_integer('nb_teachers', 250, 'Teachers in the ensemble.')
tf.flags.DEFINE_float('threshold', 300, 'Threshold for step 1 (selection).')
tf.flags.DEFINE_float('sigma1', 200, 'Sigma for step 1 (selection).')
tf.flags.DEFINE_float('sigma2', 40, 'Sigma for step 2 (argmax).')
tf.flags.DEFINE_integer('stdnt_share', 1000,
                        'Student share (last index) of the test data')
tf.flags.DEFINE_integer('lap_scale', 10,'Scale of the Laplacian noise added for privacy')#should be 10
tf.flags.DEFINE_float('eps_shift', 1e-1,
                        'Scale of the Laplacian noise added for privacy')
tf.flags.DEFINE_float('delta_shift', 1e-6,
                        'Scale of the Laplacian noise added for privacy')
tf.flags.DEFINE_boolean('save_labels', False,
                        'Dump numpy arrays of labels and clean teacher votes')
tf.flags.DEFINE_boolean('deeper', False, 'Activate deeper CNN model')
tf.flags.DEFINE_boolean('cov_shift', True, 'cov_shift instead of label shift')

def convert_vat(idx, result):

  student_file_name = FLAGS.data + 'PCA_student' + FLAGS.dataset + '.pkl'
  f = open(student_file_name, 'rb')
  log ={}
  gt = pickle.load(f)
  gt_test_data = gt['data'].reshape([-1,32*32*3])
  gt_test_label = gt['label']
  train_data = np.delete(gt_test_data,idx, axis=0)
  train_label = np.delete(gt_test_label,idx,axis=0)
  log['test_data'] = gt_test_data
  log['test_label'] = gt_test_label
  log['train_data'] = train_data
  log['train_label'] = train_label
  log['labeled_data'] =gt_test_data[idx]
  log['labeled_label'] = result
  file_vat = "../../vat_tf/log/"+FLAGS.dataset+'_query='+str(len(result))+'.pkl'
  with open(file_vat,'wb') as f:
    pickle.dump(log, f)

def gaussian(nb_labels,clean_votes,shift_idx):

  # Sample independent Laplacian noise for each class
  max_list = np.max(clean_votes, axis=1)

  for i in range(len(clean_votes)):
    max_list[i]+= FLAGS.sigma1*np.random.normal()

  idx_keep = np.where(max_list >FLAGS.threshold)
  idx_keep = np.intersect1d(idx_keep,shift_idx)
  label_release = clean_votes[idx_keep]
  result = np.zeros(len(idx_keep[0]))
  for idx, i in enumerate(label_release):
    for item in xrange(nb_labels):
      label_release[idx,item] +=  FLAGS.sigma2*np.random.normal()

    # Result is the most frequent label
    result[idx] = np.argmax(label_release[idx])

  # Cast labels to np.int32 for compatibility with deep_cnn.py feed dictionaries
  result = np.asarray(result, dtype=np.int32)
  convert_vat(idx_keep, result)
  limit = len(result)

  return (idx_keep[0][:limit],), result[:limit]



def prepare_student_data(dataset, nb_teachers,shift_idx,nb_q=None):
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
  if dataset == 'svhn':
    test_data, test_labels = input.ld_svhn(test_only=True)
  elif dataset == 'cifar10':
    test_data, test_labels = input.ld_cifar10(test_only=True)
  elif dataset == 'mnist':
    test_data, test_labels = input.ld_mnist(test_only=True)
  elif dataset == 'adult':
    test_data, test_labels = input.ld_adult(test_only = True)
  else:
    print("Check value of dataset flag")
    return False

  if nb_q !=None:
      shift_idx = np.random.choice(shift_idx, nb_q)

  # Prepare filepath for numpy dump of clean votes
  filepath = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_student_clean_votes' + str(
    FLAGS.lap_scale) + '.npy'  # NOLINT(long-line)

  if os.path.exists(filepath):

    with open(filepath,'rb')as f:
      clean_votes = np.load(f)
      keep_idx, result = gaussian(FLAGS.nb_labels, clean_votes,shift_idx)

      precision_true = metrics.accuracy(result, test_labels[keep_idx])
      print('number of idx={} precision_true from gaussian for shift data={}'.format(len(keep_idx[0]), precision_true))
      return keep_idx, test_data[keep_idx], result
  print('not find file for clean student vote')


def shift_student(stdnt_data, stdnt_labels):
  """
  This function shift student_data with a paramenter alpha
  :param student_data:
  :return: student data with shift y
  """
  shift_position = 1
  index_out = np.where(stdnt_labels == shift_position)
  # keep ratio
  ratio = 0.01
  index_remain = index_out[0][0:int(ratio * len(index_out[0]))]
  index_keep = np.where(stdnt_labels!= shift_position)
  index_keep = index_keep[0]
  index = np.concatenate((index_remain, index_keep))
 # shift_data = student_pred[index]
  shift_label = stdnt_labels[index]
  shift_dataset = {}
 # shift_dataset['pred'] = shift_data
  shift_dataset['label'] = shift_label
  shift_dataset['shift_ratio'] = ratio
  shift_dataset['data'] = stdnt_data[index]
  shift_dataset['index'] = index
  return shift_dataset

def dir_shift(stdnt_data,  stdnt_labels, alpha):
  """
  This function produce dirichlet shift for student dataset
  :param stdnt_data:
  :param student_pred:
  :param stdnt_labels:
  :return: student data with shift parameter alpha
  """
  label = np.unique(stdnt_labels)
  alpha_array = [alpha for i in label]
  dist = np.random.dirichlet(alpha_array, 1)
  dist = dist[0]
  num_data = stdnt_data.shape[0]
  choice = np.random.choice(label, num_data,p = dist)

  index  = [len(stdnt_labels)-1] # at least one element
  for x in label:
    index_out = np.where(stdnt_labels == x)
    cur_number = np.where(choice == x)[0].shape[0]
    if cur_number == 0: # none element in this class have been chosen
        index_remain = index_out[0][:1]
    else:
        index_remain = np.random.choice(index_out[0],cur_number,replace = True)
    index = np.concatenate((index, index_remain))
  np.random.shuffle(index)
  #shift_data = student_pred[index]
  shift_label = stdnt_labels[index]
  shift_dataset = {}
  #shift_dataset['pred'] = shift_data
  shift_dataset['label'] = shift_label
  shift_dataset['alpha'] = alpha
  shift_dataset['index'] = index
  return shift_dataset
def predict_teacher(dataset, nb_teachers):
  """
  This is for obtaining the weight from student / teache, don't involve any noise
  :param dataset:  string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :param teacher: if teacher is true, then predict with training dataset, else students
  :return: out prediction based on cnn
  """
  assert input.create_dir_if_needed(FLAGS.train_dir)

  train_only = True
  test_only  = False

  # create path to save teacher predict teacher model
  filepath = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_teacher_clean_votes_label_shift' + str(
    FLAGS.lap_scale) + '.npy'
  # Load the dataset
  if dataset == 'svhn':
    test_data, test_labels = input.ld_svhn(test_only, train_only)
  elif dataset == 'cifar10':
    test_data, test_labels = input.ld_cifar10(test_only, train_only)
  elif dataset == 'mnist':
    test_data, test_labels = input.ld_mnist(test_only, train_only)
  elif dataset == 'adult':
    test_data, test_labels = input.ld_adult(test_only, train_only)
  else:
    print("Check value of dataset flag")
    return False
  if os.path.exists(filepath):
    pred_labels = np.load(filepath)
     # Print accuracy of aggregated labels
    ac_ag_labels = metrics.accuracy(pred_labels, test_labels)
    print("obtain_weight Accuracy of the aggregated labels for teachers: " + str(ac_ag_labels))
    return  pred_labels, test_labels
  else:
    print('not found teacher file')

def obtain_weight(knock, student_data, nb_teacher):
  """
  This function use pretrained model on nb_teacher to obtain the importance weight of student/teacher
  we assue the student dataset is unlabeled
  we use the whole training set as private one for teacher
  and the whole test set as public one for student

  :param teacher_data:
  :param student_data: unshift student_data
  :param nb_teacher:
  :return: an importance weight of student(y)/teacher(y)
  """
  assert input.create_dir_if_needed(FLAGS.train_dir)
  # Call helper function to prepare student data using teacher predictions

  # Unpack the student dataset

  stdnt_data, stdnt_labels = utils.load_dataset(FLAGS.dataset, test_only=True)

  if knock == False:
      shift_idx, shift_dataset = dir_shift(stdnt_data, stdnt_labels, 0.1)
  else:
      shift_dataset = shift_student(stdnt_data, stdnt_labels)
      # stdnt_pred_total means put all test dataset in, not consider shift here
  shift_idx =shift_dataset['index']
  shift_idx, stdnt_test, stdnt_pred = prepare_student_data(FLAGS.dataset, FLAGS.nb_teachers,shift_idx)

  shift_dataset['pred'] = stdnt_pred
  shift_dataset['index'] = shift_idx
  shift_dataset['label'] - stdnt_labels[shift_idx]
  shift_dataset['data'] = stdnt_data[shift_idx]
  # check shape here
  # students' prediction after shift

  teacher_pred, teacher_test = predict_teacher(FLAGS.dataset, FLAGS.nb_teachers)
  dis_t = np.zeros(FLAGS.nb_labels)
  dis_s = np.zeros(FLAGS.nb_labels)
  for i in range(FLAGS.nb_labels):
      dis_t[i] = np.sum(teacher_test == i)
      dis_s[i] = np.sum(shift_dataset['label'] == i)

  dis_t = dis_t / len(teacher_test)
  dis_s = dis_s / len(shift_dataset['label'])
  print('teacher distribution = {}'.format(dis_t))
  print('shift student distribution= {}'.format(dis_s))

  num_class = FLAGS.nb_labels
  # mu is average predict in student
  mu = np.zeros(num_class)
  for ind in range(num_class):
      mu[ind] = np.sum(stdnt_pred == ind)
  mu = mu / len(stdnt_pred)
  cov = np.zeros([num_class, num_class])
  for index, x in enumerate(teacher_pred):
      cov[x, teacher_test[index]] += 1
  cov = cov / len(teacher_test)
  np.reciprocal(cov, cov)
  w = np.dot(cov, mu)
  inverse_w = np.reciprocal(w)
  return shift_dataset, inverse_w


def train_student(dataset, nb_teachers, shift_dataset,inverse_w=None, weight = True):
  """
  This function trains a student using predictions made by an ensemble of
  teachers. The student and teacher models are trained using the same
  neural network architecture.
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :param weight: whether this is an importance weight sampling
  :return: True if student training went well
  """
  assert input.create_dir_if_needed(FLAGS.train_dir)

  # Call helper function to prepare student data using teacher predictions

  stdnt_data = shift_dataset['data']
  stdnt_labels = shift_dataset['pred']

  print('number for deep is {}'.format(len(stdnt_labels)))

  if FLAGS.deeper:
    ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student_deeper.ckpt' #NOLINT(long-line)
  else:
    ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student.ckpt'  # NOLINT(long-line)

  if FLAGS.cov_shift == True:
    """
       need to compute the weight for student
       curve weight into some bound, in case the weight is too large
    """
    weights = inverse_w
  else:
    print('len of shift data'.format(len(shift_dataset['data'])))
    weights = np.zeros(len(stdnt_data))
    print('len of weight={} len of labels= {} '.format(len(weights), len(stdnt_labels)))
    for i, x in enumerate(weights):
      weights[i] = np.float32(inverse_w[stdnt_labels[i]])

  if weight == True:
    assert deep_cnn.train(stdnt_data, stdnt_labels, ckpt_path, weights= weights)
  else:
    deep_cnn.train(stdnt_data, stdnt_labels, ckpt_path)
  # Compute final checkpoint name for student (with max number of steps)
  ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)
  if dataset == 'adult':
    private_data, private_labels = input.ld_adult(test_only = False, train_only= True)
  elif dataset =='mnist':
    private_data, private_labels = input.ld_mnist(test_only = False, train_only = True)
  elif dataset =="svhn":
    private_data, private_labels = input.ld_svhn(test_only=False, train_only=True)
  # Compute student label predictions on remaining chunk of test set
  teacher_preds = deep_cnn.softmax_preds(private_data, ckpt_path_final)
  student_preds =  deep_cnn.softmax_preds(stdnt_data, ckpt_path_final)
  # Compute teacher accuracy
  precision_t = metrics.accuracy(teacher_preds, private_labels)
  precision_s  = metrics.accuracy(student_preds, stdnt_labels)

  precision_true = metrics.accuracy(student_preds, shift_dataset['label'])
  print('Precision of teacher after training:{} student={} true precision for student {}'.format(precision_t, precision_s,precision_true))

  return precision_t, precision_s
def main(argv=None): # pylint: disable=unused-argument

  stdnt_data, stdnt_labels = utils.load_dataset(FLAGS.dataset, test_only=True)

    # Run student training according to values specified in flags

  wei_precision_t = []
  wei_precision_s = []
  non_precision_t = []
  non_precision_s = []
  if FLAGS.cov_shift == True:
    cov_shift.pca_transform(FLAGS.dataset, FLAGS)
    theta = cov_shift.cov_logistic(FLAGS) # theta is the parameter in logistic, used for importance weight
    import pickle
    student_file_name = FLAGS.data + 'PCA_student' + FLAGS.dataset + '.pkl'
    f = open(student_file_name, 'rb')
    shift_dataset = pickle.load(f)
    shift_idx, stdnt_test, stdnt_pred = prepare_student_data(FLAGS.dataset, FLAGS.nb_teachers, shift_dataset['index'])
    shift_dataset['pred'] = stdnt_pred
    shift_dataset['index'] = shift_idx
    shift_dataset['label'] = stdnt_labels[shift_idx]
    shift_dataset['data'] = stdnt_data[shift_idx]

    theta_path = FLAGS.dataset +'theta.pkl'
    f = open(theta_path, 'rb')
    #pickle.dump(theta, f)

    theta = pickle.load(f)
    s1 = np.sum(theta)
    theta = len(theta)/s1*theta
    theta = np.ravel(theta)
    print('theta max={} mean={} min = {}'.format(np.max(theta), np.mean(theta),np.min(theta)))
    for weight in [True, False]:
      ac_t, ac_s = train_student(FLAGS.dataset, FLAGS.nb_teachers,shift_dataset,inverse_w=theta,weight=weight)
      ac_t = round(ac_t, 2)
      ac_s = round(ac_s, 2)
      print('weight={} ac_t={} ac_s={}'.format(weight, ac_t, ac_s))
      if weight == True:
        wei_precision_t.append(ac_t)
        wei_precision_s.append(ac_s)
      else:
        non_precision_t.append(ac_t)
        non_precision_s.append(ac_s)
  else:
    for knock in [False]:
      # here 2 is the number of teacher to estimate weight
      # knock is False refer to derichlet shift
      # We average the accuracy results for 10 time
      for idx in range(5):
        shift_dataset, inverse_w = obtain_weight(knock, FLAGS.dataset, FLAGS.nb_teachers)
        print('******* inverse_weight={}'.format(inverse_w))
        sum = np.sum(inverse_w)
        for iidx, x in enumerate(inverse_w):
          if x > 30:
            inverse_w[iidx] = 30
            print('max_value={}'.format(x))
        inverse_w = [x / sum * 10 for x in inverse_w]
        print('inverse weight = {}'.format(inverse_w))
        for weight in [True, False]:
          ac_t, ac_s = train_student(FLAGS.dataset, FLAGS.nb_teachers, shift_dataset,inverse_w=inverse_w,weight=weight)
          ac_t = round(ac_t, 2)
          ac_s = round(ac_s, 2)
          if weight == True:
            wei_precision_t.append(ac_t)
            wei_precision_s.append(ac_s)
          else:
            non_precision_t.append(ac_t)
            non_precision_s.append(ac_s)

    ckpt_path = FLAGS.train_dir + '/' + str(FLAGS.dataset) + 'result.txt'

    with open(ckpt_path, 'w') as f:
      if FLAGS.cov_shift == True:
        f.write('PCA methods:\n')
        f.write('PATE2={} thresh= {} sigma1={} sigma2={}'.format(FLAGS.PATE2,FLAGS.threshold, FLAGS.sigma1,FLAGS.sigma2))
        f.write('\n')
       # f.write('total query={} answer query={}'.format(len(theta),num_query))
      elif knock == True:
        f.write('knock out methods:\n')
      else:
        f.write('dirshlet methods:\n' + str(0.03))
      f.write('weighted prediction with first teacher then students\n')
      for idx, item in enumerate(wei_precision_t):
        f.write('teacher:' + str(wei_precision_t[idx]) + ' student' + str(wei_precision_s[idx]))
      f.write('\n')
      f.write('nonweight prediction\n')
      for idx, item in enumerate(non_precision_t):
        f.write('teacher:' + str(non_precision_t[idx]) + ' student' + str(non_precision_s[idx]))

      f.write('\n')


if __name__ == '__main__':

  tf.app.run()
