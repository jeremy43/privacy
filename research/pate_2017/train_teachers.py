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

import tensorflow as tf
import deep_cnn
import input
import metrics
import pickle
tf.flags.DEFINE_string('dataset', 'mnist', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')

tf.flags.DEFINE_string('data_dir','/Users/yuqing/github_proj/privacy/research/','Temporary storage')
tf.flags.DEFINE_string('train_dir','/Users/yuqing/github_proj/privacy/research/model',
                     'Where model ckpt are saved')
tf.flags.DEFINE_string('data','/Users/yuqing/github_proj/privacy/research/data', 'where pca data are saved ')

tf.flags.DEFINE_integer('max_steps', 3000, 'Number of training steps to run.')
tf.flags.DEFINE_integer('nb_teachers', 2, 'Teachers in the ensemble.')
tf.flags.DEFINE_integer('teacher_id', 0, 'ID of teacher being trained.')
tf.flags.DEFINE_boolean('cov_shift', True, 'cov_shift instead of label shift')
tf.flags.DEFINE_boolean('deeper', False, 'Activate deeper CNN model')

FLAGS = tf.flags.FLAGS


def train_teacher(FLAGS, dataset, nb_teachers, teacher_id):
  """
  This function trains a teacher (teacher id) among an ensemble of nb_teachers
  models for the dataset specified.
  :param dataset: string corresponding to dataset (svhn, cifar10)
  :param nb_teachers: total number of teachers in the ensemble
  :param teacher_id: id of the teacher being trained
  :return: True if everything went well
  """
  # If working directories do not exist, create them
  assert input.create_dir_if_needed(FLAGS.data_dir)
  assert input.create_dir_if_needed(FLAGS.train_dir)

  # Load the dataset
  if dataset == 'svhn':
    train_data,train_labels,test_data,test_labels = input.ld_svhn(extended=True)
  elif dataset == 'cifar10':
    train_data, train_labels, test_data, test_labels = input.ld_cifar10()
  elif dataset == 'mnist':
    train_data, train_labels, test_data, test_labels = input.ld_mnist()
  else:
    print("Check value of dataset flag")
    return False
  if FLAGS.cov_shift == True:
    teacher_file_name = FLAGS.data + 'PCA_teacher' + FLAGS.dataset + '.pkl'
    student_file_name = FLAGS.data + 'PCA_student' + FLAGS.dataset + '.pkl'
    f = open(teacher_file_name, 'rb')
    train_data = pickle.load(f)
    f = open(student_file_name, 'rb')
    test_data = pickle.load(f)
  # Retrieve subset of data for this teacher
  data, labels = input.partition_dataset(train_data,
                                         train_labels,
                                         nb_teachers,
                                         teacher_id)

  print("Length of training data: " + str(len(labels)))

  # Define teacher checkpoint filename and full path
  if FLAGS.deeper:
    filename = str(nb_teachers) + 'pca_teachers_' + str(teacher_id) + '_deep.ckpt'
  else:
    filename = str(nb_teachers) + 'pca_teachers_' + str(teacher_id) + '.ckpt'
  ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + filename

  # Perform teacher training
  assert deep_cnn.train(data, labels, ckpt_path)

  # Append final step value to checkpoint for evaluation
  ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)

  # Retrieve teacher probability estimates on the test data
  teacher_preds = deep_cnn.softmax_preds(test_data, ckpt_path_final)

  # Compute teacher accuracy
  precision = metrics.accuracy(teacher_preds, test_labels)
  print('Precision of teacher after training: ' + str(precision))

  return True


def main(argv=None):  # pylint: disable=unused-argument
  # Make a call to train_teachers with values specified in flags
  for id in range(FLAGS.nb_teachers):
    assert train_teacher(FLAGS, FLAGS.dataset, FLAGS.nb_teachers, id)

if __name__ == '__main__':
  tf.app.run()
