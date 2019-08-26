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
import tensorflow as tf
import numpy as np
import input

def batch_indices(batch_nb, data_length, batch_size):
  """
  This helper function computes a batch start and end index
  :param batch_nb: the batch number
  :param data_length: the total length of the data being parsed by batches
  :param batch_size: the number of inputs in each batch
  :return: pair of (start, end) indices
  """
  # Batch start and end index
  start = int(batch_nb * batch_size)
  end = int((batch_nb + 1) * batch_size)

  # When there are not enough inputs left, we reuse some to complete the batch
  if end > data_length:
    shift = end - data_length
    start -= shift
    end -= shift

  return start, end

def save_file(path, file):
  with tf.gfile.Open(path, mode='w') as file_obj:
    np.save(file_obj, file)

def load_dataset(dataset, test_only=False, train_only=False):

  if dataset == 'svhn':
    test_data, test_labels = input.ld_svhn(test_only=test_only)
    return test_data, test_labels
  elif dataset == 'cifar10':
    test_data, test_labels = input.ld_cifar10(test_only=test_only)
  elif dataset == 'mnist':
    test_data, test_labels = input.ld_mnist(test_only=test_only)
  elif dataset == 'adult':
    test_data, test_labels = input.ld_adult(test_only = test_only)
  else:
    print("Check value of dataset flag")
  return test_data, test_labels

def create_path(FLAGS,dataset, nb_teachers):
  assert input.create_dir_if_needed(FLAGS.train_dir)
  gau_filepath = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_student_votes_sigma1:' + str(
    FLAGS.sigma1) + '_sigma2:' + str(FLAGS.sigma2) + '.npy'  # NOLINT(long-line)

  # Prepare filepath for numpy dump of clean votes
  filepath = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_student_clean_votes_label_shift' + str(
    FLAGS.lap_scale) + '.npy'  # NOLINT(long-line)

  # Prepare filepath for numpy dump of clean labels
  filepath_labels = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_teachers_labels_' + str(
    FLAGS.lap_scale) + '.npy'  # NOLINT(long-line)

  return gau_filepath,filepath,filepath_labels
