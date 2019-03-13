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

import numpy as np
from six.moves import xrange
import tensorflow as tf

import aggregation
import deep_cnn
import input
import metrics

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('dataset', 'mnist', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')

tf.flags.DEFINE_string('data_dir','/tmp','Temporary storage')
tf.flags.DEFINE_string('train_dir','/tmp/train_dir','Where model chkpt are saved')
tf.flags.DEFINE_string('teachers_dir','/tmp/train_dir',
                       'Directory where teachers checkpoints are stored.')

tf.flags.DEFINE_integer('teachers_max_steps', 3000,
                        'Number of steps teachers were ran.')
tf.flags.DEFINE_integer('max_steps', 3000, 'Number of steps to run student.')
tf.flags.DEFINE_integer('nb_teachers', 10, 'Teachers in the ensemble.')
tf.flags.DEFINE_integer('stdnt_share', 1000,
                        'Student share (last index) of the test data')
tf.flags.DEFINE_integer('lap_scale', 10,
                        'Scale of the Laplacian noise added for privacy')
tf.flags.DEFINE_boolean('save_labels', False,
                        'Dump numpy arrays of labels and clean teacher votes')
tf.flags.DEFINE_boolean('deeper', False, 'Activate deeper CNN model')


def ensemble_preds(dataset, nb_teachers, stdnt_data):
  """
  Given a dataset, a number of teachers, and some input data, this helper
  function queries each teacher for predictions on the data and returns
  all predictions in a single array. (That can then be aggregated into
  one single prediction per input using aggregation.py (cf. function
  prepare_student_data() below)
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :param stdnt_data: unlabeled student training data
  :return: 3d array (teacher id, sample id, probability per class)
  """

  # Compute shape of array that will hold probabilities produced by each
  # teacher, for each training point, and each output class
  result_shape = (nb_teachers, len(stdnt_data), FLAGS.nb_labels)

  # Create array that will hold result
  result = np.zeros(result_shape, dtype=np.float32)

  # Get predictions from each teacher #should start at 0
  for teacher_id in xrange(nb_teachers):
    # Compute path of checkpoint file for teacher model with ID teacher_id
    if FLAGS.deeper:
      ckpt_path = FLAGS.teachers_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_teachers_' + str(teacher_id) + '_deep.ckpt-' + str(FLAGS.teachers_max_steps - 1) #NOLINT(long-line)
    else:
      ckpt_path = FLAGS.teachers_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_teachers_' + str(teacher_id) + '.ckpt-' + str(FLAGS.teachers_max_steps - 1)  # NOLINT(long-line)

    # Get predictions on our training data and store in result array
    result[teacher_id] = deep_cnn.softmax_preds(stdnt_data, ckpt_path)

    # This can take a while when there are a lot of teachers so output status
    print("Computed Teacher " + str(teacher_id) + " softmax predictions")

  return result

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
  print("Accuracy of the aggregated labels: " + str(ac_ag_labels))
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

    stdnt_data = shift_data

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
  ac_ag_labels = metrics.accuracy(stdnt_labels, test_labels[:FLAGS.stdnt_share])
  print("Accuracy of the aggregated labels: " + str(ac_ag_labels))



  if save:
    # Prepare filepath for numpy dump of labels produced by noisy aggregation
    filepath = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_student_labels_lap_' + str(FLAGS.lap_scale) + '.npy' #NOLINT(long-line)

    # Dump student noisy labels array
    with tf.gfile.Open(filepath, mode='w') as file_obj:
      np.save(file_obj, stdnt_labels)

  return stdnt_data, stdnt_labels

def shift_student(student_pred, stdnt_labels):
  """
  This function shift student_data with a paramenter alpha
  :param student_data:
  :return: student data with shift y
  """
  shift_position = 5
  index_out = np.where(stdnt_labels == shift_position)
  # keep ratio
  ratio = 0.5
  index_remain = index_out[0][0:int(ratio * len(index_out[0]))]
  index_keep = np.where(stdnt_labels!= 5)
  index_keep = index_keep[0]
  index = np.concatenate((index_remain, index_keep))
  shift_data = student_pred[index]
  shift_label = stdnt_labels[index]
  shift_dataset = {}
  shift_dataset['pred'] = shift_data
  shift_dataset['label'] = shift_label
  shift_dataset['shift_ratio'] = ratio
  return shift_dataset
def obtain_weight(student_data, nb_teacher):
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
  _, teacher_pred, teacher_test = predict_data(student_data, nb_teacher, teacher = True)
  # Unpack the student dataset
  stdnt_data, stdnt_pred, stdnt_test = predict_data(student_data, nb_teacher, teacher=False)
  shift_dataset= shift_student(stdnt_pred, stdnt_test)
  shift_dataset['data'] = stdnt_data
  #students' prediction after shift
  stdnt_pred = shift_dataset['pred']
  stdnt_labels = shift_dataset['label']
  # model_path = FLAGS.train_dir + '/' + 'mnist_250_teachers_1.ckpt-2999'
  # Compute student label predictions
  #
  # student_preds = deep_cnn.softmax_preds(stdnt_data, model_path)
  # # Here we use the test dataset of students to estimate teacher, since they are from same distution
  # student_preds =np.argmax(student_preds, axis = 1)
  # teacher_estimate = deep_cnn.softmax_preds(stdnt_test_data, model_path)
  # teacher_estimate = np.argmax(teacher_estimate, axis=1)
  num_class = np.max(stdnt_test) +1
  # mu is average predict in student
  mu = np.zeros(num_class)
  for ind in range(num_class):
    mu[ind] = np.sum(stdnt_pred==ind)
  mu = mu /len(stdnt_pred)
  cov = np.zeros([num_class, num_class])
  for index, x in enumerate(teacher_pred):
    cov[x, teacher_test[index]] += 1
  cov = cov / len(teacher_test)
  np.reciprocal(mu, mu)
  inverse_w = np.dot(cov, mu)
  return shift_dataset, inverse_w


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

  # Call helper function to prepare student data using teacher predictions
  stdnt_data, stdnt_labels= prepare_student_data(dataset, nb_teachers, save=True, shift_dataset = shift_dataset['data'])

  # Unpack the student dataset, here stdnt_labels are already the ensemble noisy version
  if shift_dataset is not None:
    stdnt_data = shift_dataset['data']
    stdnt_labels = shift_dataset['label']
  # Prepare checkpoint filename and path
  if FLAGS.deeper:
    ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student_deeper.ckpt' #NOLINT(long-line)
  else:
    ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + str(nb_teachers) + '_student.ckpt'  # NOLINT(long-line)

  # Start student training
  weights = np.zeros(len(stdnt_data))

  for i, x in enumerate(weights):
    weights[i] = np.float32(inverse_w[stdnt_labels[i]])
  assert deep_cnn.train(stdnt_data, stdnt_labels, ckpt_path, weights= weights)

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

def main(argv=None): # pylint: disable=unused-argument
  # Run student training according to values specified in flags
  shift_dataset, inverse_w = obtain_weight(FLAGS.dataset, 1)
  assert train_student(FLAGS.dataset, FLAGS.nb_teachers, inverse_w, shift_dataset)

if __name__ == '__main__':
  tf.app.run()
