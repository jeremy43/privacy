# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""PrivacyLedger class for keeping a record of private queries.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from privacy.analysis import tensor_buffer
from privacy.optimizers import dp_query

nest = tf.contrib.framework.nest

SampleEntry = collections.namedtuple(  # pylint: disable=invalid-name
    'SampleEntry', ['population_size', 'selection_probability', 'queries'])

GaussianSumQueryEntry = collections.namedtuple(  # pylint: disable=invalid-name
    'GaussianSumQueryEntry', ['l2_norm_bound', 'noise_stddev'])


class PrivacyLedger(object):
  """Class for keeping a record of private queries.

  The PrivacyLedger keeps a record of all queries executed over a given dataset
  for the purpose of computing privacy guarantees.
  """

  def __init__(
      self,
      population_size,
      selection_probability,
      max_samples,
      max_queries):
    """Initialize the PrivacyLedger.

    Args:
      population_size: An integer (may be variable) specifying the size of the
        population.
      selection_probability: A float (may be variable) specifying the
        probability each record is included in a sample.
      max_samples: The maximum number of samples. An exception is thrown if
        more than this many samples are recorded.
      max_queries: The maximum number of queries. An exception is thrown if
        more than this many queries are recorded.
    """
    self._population_size = population_size
    self._selection_probability = selection_probability

    # The query buffer stores rows corresponding to GaussianSumQueryEntries.
    self._query_buffer = tensor_buffer.TensorBuffer(
        max_queries, [3], tf.float32, 'query')
    self._sample_var = tf.Variable(
        initial_value=tf.zeros([3]), trainable=False, name='sample')

    # The sample buffer stores rows corresponding to SampleEntries.
    self._sample_buffer = tensor_buffer.TensorBuffer(
        max_samples, [3], tf.float32, 'sample')
    self._sample_count = tf.Variable(
        initial_value=0.0, trainable=False, name='sample_count')
    self._query_count = tf.Variable(
        initial_value=0.0, trainable=False, name='query_count')
    try:
      # Newer versions of TF
      self._cs = tf.CriticalSection()
    except AttributeError:
      # Older versions of TF
      self._cs = tf.contrib.framework.CriticalSection()

  def record_sum_query(self, l2_norm_bound, noise_stddev):
    """Records that a query was issued.

    Args:
      l2_norm_bound: The maximum l2 norm of the tensor group in the query.
      noise_stddev: The standard deviation of the noise applied to the sum.

    Returns:
      An operation recording the sum query to the ledger.
    """
    def _do_record_query():
      with tf.control_dependencies([
          tf.assign(self._query_count, self._query_count + 1)]):
        return self._query_buffer.append(
            [self._sample_count, l2_norm_bound, noise_stddev])

    return self._cs.execute(_do_record_query)

  def finalize_sample(self):
    """Finalizes sample and records sample ledger entry."""
    with tf.control_dependencies([
        tf.assign(
            self._sample_var,
            [self._population_size,
             self._selection_probability,
             self._query_count])]):
      with tf.control_dependencies([
          tf.assign(self._sample_count, self._sample_count + 1),
          tf.assign(self._query_count, 0)]):
        return self._sample_buffer.append(self._sample_var)

  def _format_ledger(self, sample_array, query_array):
    """Converts underlying representation into a list of SampleEntries."""
    samples = []
    query_pos = 0
    sample_pos = 0
    for sample in sample_array:
      num_queries = int(sample[2])
      queries = []
      for _ in range(num_queries):
        query = query_array[query_pos]
        assert int(query[0]) == sample_pos
        queries.append(GaussianSumQueryEntry(*query[1:]))
        query_pos += 1
      samples.append(SampleEntry(sample[0], sample[1], queries))
      sample_pos += 1
    return samples

  def get_formatted_ledger(self, sess):
    """Gets the formatted query ledger.

    Args:
      sess: The tensorflow session in which the ledger was created.

    Returns:
      The query ledger as a list of SampleEntries.
    """
    sample_array = sess.run(self._sample_buffer.values)
    query_array = sess.run(self._query_buffer.values)

    return self._format_ledger(sample_array, query_array)

  def get_formatted_ledger_eager(self):
    """Gets the formatted query ledger.

    Returns:
      The query ledger as a list of SampleEntries.
    """
    sample_array = self._sample_buffer.values.numpy()
    query_array = self._query_buffer.values.numpy()

    return self._format_ledger(sample_array, query_array)


class QueryWithLedger(dp_query.DPQuery):
  """A class for DP queries that record events to a PrivacyLedger.

  QueryWithLedger should be the top-level query in a structure of queries that
  may include sum queries, nested queries, etc. It should simply wrap another
  query and contain a reference to the ledger. Any contained queries (including
  those contained in the leaves of a nested query) should also contain a
  reference to the same ledger object.

  For example usage, see privacy_ledger_test.py.
  """

  def __init__(self, query, ledger):
    """Initializes the QueryWithLedger.

    Args:
      query: The query whose events should be recorded to the ledger. Any
        subqueries (including those in the leaves of a nested query) should
        also contain a reference to the same ledger given here.
      ledger: A PrivacyLedger to which privacy events should be recorded.
    """
    self._query = query
    self._ledger = ledger

  def initial_global_state(self):
    """See base class."""
    return self._query.initial_global_state()

  def derive_sample_params(self, global_state):
    """See base class."""
    return self._query.derive_sample_params(global_state)

  def initial_sample_state(self, global_state, tensors):
    """See base class."""
    return self._query.initial_sample_state(global_state, tensors)

  def accumulate_record(self, params, sample_state, record):
    """See base class."""
    return self._query.accumulate_record(params, sample_state, record)

  def get_noised_result(self, sample_state, global_state):
    """Ensures sample is recorded to the ledger and returns noised result."""
    with tf.control_dependencies(nest.flatten(sample_state)):
      with tf.control_dependencies([self._ledger.finalize_sample()]):
        return self._query.get_noised_result(sample_state, global_state)
