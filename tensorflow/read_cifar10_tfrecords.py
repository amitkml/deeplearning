import argparse
import os
import sys

import tarfile
from six.moves import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""CIFAR-10 data set.
See http://www.cs.toronto.edu/~kriz/cifar.html.
"""

### https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator
import os

import tensorflow as tf
from keras.callbacks import Callback



HEIGHT = 32
WIDTH = 32
DEPTH = 3
SHUFFLE_BUFFER=512
BATCH_SIZE=512
HEIGHT = 32
WIDTH = 32
DEPTH = 3
SUM_OF_ALL_DATASAMPLES = 50000

def preprocess(subset,image):
    # """Preprocess a single image in [height, width, depth] layout."""
  if subset == 'train' :

    # Pad 4 pixels on each dimension of feature map, done in mini-batch
    image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
    image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
    image = tf.image.random_flip_left_right(image)
  return image


  def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)

    # Custom preprocessing.
    image = preprocess('training',image)

    return image, label

  def valparser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)

    # Custom preprocessing.
    # image = preprocess('training',image)

    return image, label

  # """Read the images and labels from 'filenames'."""
  # Repeat infinitely.
def make_batch(filename, batch_size):
  dataset = tf.data.TFRecordDataset(filename).repeat()
  # Parse records.
  dataset = dataset.map(
      parser, num_parallel_calls=batch_size)

  # # Potentially shuffle records.
  # if subset == 'train':
  #   min_queue_examples = int(
  #         Cifar10DataSet.num_examples_per_epoch(self.subset) * 0.4)
  #     # Ensure that the capacity is sufficiently large to provide good random
  #     # shuffling.
  # dataset = dataset.shuffle(SHUFFLE_BUFFER )

    # Batch it up.
  dataset = dataset.batch(BATCH_SIZE)
  iterator = dataset.make_one_shot_iterator()
  image_batch, label_batch = iterator.get_next()
   # Create a one hot array for your labels
  label_batch = tf.one_hot(label_batch, num_classes)
  return image_batch, label_batch

  # """Read the images and labels from 'filenames'."""
  # Repeat infinitely.
def val_make_batch(filename, batch_size):
  dataset = tf.data.TFRecordDataset(filename).repeat()
  # Parse records.
  dataset = dataset.map(
      valparser, num_parallel_calls=batch_size)

  # # Potentially shuffle records.
  # if subset == 'train':
  #   min_queue_examples = int(
  #         Cifar10DataSet.num_examples_per_epoch(self.subset) * 0.4)
  #     # Ensure that the capacity is sufficiently large to provide good random
  #     # shuffling.
  # dataset = dataset.shuffle(SHUFFLE_BUFFER )

    # Batch it up.
  dataset = dataset.batch(BATCH_SIZE)
  iterator = dataset.make_one_shot_iterator()
  image_batch, label_batch = iterator.get_next()
   # Create a one hot array for your labels
  label_batch = tf.one_hot(label_batch, num_classes)
  return image_batch, label_batch

class EvaluateInputTensor(Callback):
    """ Validate a model which does not expect external numpy data during training.

    Keras does not expect external numpy data at training time, and thus cannot
    accept numpy arrays for validation when all of a Keras Model's
    `Input(input_tensor)` layers are provided an  `input_tensor` parameter,
    and the call to `Model.compile(target_tensors)` defines all `target_tensors`.
    Instead, create a second model for validation which is also configured
    with input tensors and add it to the `EvaluateInputTensor` callback
    to perform validation.

    It is recommended that this callback be the first in the list of callbacks
    because it defines the validation variables required by many other callbacks,
    and Callbacks are made in order.

    # Arguments
        model: Keras model on which to call model.evaluate().
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
    """

    def __init__(self, test_model, steps, metrics_prefix='val', verbose=1):
        # parameter of callbacks passed during initialization
        # pass evalation mode directly
        super(EvaluateInputTensor, self).__init__()
        self.val_model = model
        self.num_steps = steps
        self.verbose = verbose
        self.metrics_prefix = metrics_prefix

    def on_epoch_end(self, epoch, logs={}):
        self.val_model.set_weights(self.model.get_weights())
        results = self.val_model.evaluate(None, None, steps=int(self.num_steps),
                                          verbose=self.verbose)
        metrics_str = '\n'
        for result, name in zip(results, self.val_model.metrics_names):
            metric_name = self.metrics_prefix + '_' + name
            logs[metric_name] = result
            if self.verbose > 0:
                metrics_str = metrics_str + metric_name + ': ' + str(result) + ' '

        if self.verbose > 0:
            print(metrics_str)

# class Cifar10DataSet(object):
#   """Cifar10 data set.
#   Described by http://www.cs.toronto.edu/~kriz/cifar.html.
#   """
#
#   def __init__(self, data_dir, subset='train', use_distortion=True):
#     self.data_dir = data_dir
#     self.subset = subset
#     self.use_distortion = use_distortion
#
#   def get_filenames(self):
#     if self.subset in ['train', 'validation', 'eval']:
#       return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
#     else:
#       raise ValueError('Invalid data subset "%s"' % self.subset)
#
#   def parser(self, serialized_example):
#     """Parses a single tf.Example into image and label tensors."""
#     # Dimensions of the images in the CIFAR-10 dataset.
#     # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
#     # input format.
#     features = tf.parse_single_example(
#         serialized_example,
#         features={
#             'image': tf.FixedLenFeature([], tf.string),
#             'label': tf.FixedLenFeature([], tf.int64),
#         })
#     image = tf.decode_raw(features['image'], tf.uint8)
#     image.set_shape([DEPTH * HEIGHT * WIDTH])
#
#     # Reshape from [depth * height * width] to [depth, height, width].
#     image = tf.cast(
#         tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
#         tf.float32)
#     label = tf.cast(features['label'], tf.int32)
#
#     # Custom preprocessing.
#     image = self.preprocess(image)
#
#     return image, label
#
#   def make_batch(self, batch_size):
#     """Read the images and labels from 'filenames'."""
#     filenames = self.get_filenames()
#     # Repeat infinitely.
#     dataset = tf.data.TFRecordDataset(filenames).repeat()
#
#     # Parse records.
#     dataset = dataset.map(
#         self.parser, num_parallel_calls=batch_size)
#
#     # Potentially shuffle records.
#     if self.subset == 'train':
#       min_queue_examples = int(
#           Cifar10DataSet.num_examples_per_epoch(self.subset) * 0.4)
#       # Ensure that the capacity is sufficiently large to provide good random
#       # shuffling.
#       dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)
#
#     # Batch it up.
#     dataset = dataset.batch(batch_size)
#     iterator = dataset.make_one_shot_iterator()
#     image_batch, label_batch = iterator.get_next()
#
#     return image_batch, label_batch
#
#   def preprocess(self, image):
#     """Preprocess a single image in [height, width, depth] layout."""
#     if self.subset == 'train' and self.use_distortion:
#       # Pad 4 pixels on each dimension of feature map, done in mini-batch
#       image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
#       image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
#       image = tf.image.random_flip_left_right(image)
#     return image
#
#   @staticmethod
#   def num_examples_per_epoch(subset='train'):
#     if subset == 'train':
#       return 45000
#     elif subset == 'validation':
#       return 5000
#     elif subset == 'eval':
#       return 10000
#     else:
#       raise ValueError('Invalid data subset "%s"' % subset)
