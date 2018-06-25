# Copyright 2015 Paul Balanca. All Rights Reserved.
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
"""Converts DICOM data to TFRecords file format with Example protos.

The raw DICOM data set is expected to reside in JPEG files located in the
directory 'images'. Similarly, bounding box annotations are supposed to be
stored in the 'labels'
"""
import os
import os.path
import sys
import random

import numpy as np
import tensorflow as tf

from dataset.dataset_utils import int64_feature, float_feature, bytes_feature
from dataset.dicom_common import DICOM_LABELS

DEFAULT_IMAGE_DIR = 'images/'
DEFAULT_LABEL_DIR = 'labels/'

# The number of images in the validation set.
NUM_VALIDATION = 3002  # about 10% of the entire dataset
NUM_TEST = 3001 # about 10% of the entire dataset

# Seed for repeatability.
# RANDOM_SEED = 123
# RANDOM_SEED = 111 # first cross-validation
RANDOM_SEED = 79


def _jpeg_image_shape(image_data, sess, decoded_jpeg, inputs):
    rimg = sess.run(decoded_jpeg, feed_dict={inputs: image_data})
    assert len(rimg.shape) ==3
    assert rimg.shape[2] == 3
    return rimg.shape


def _process_image(directory, name, f_jpeg_image_shape,
                   image_dir=DEFAULT_IMAGE_DIR, label_dir=DEFAULT_LABEL_DIR):
    """Process a image and annotation file.

    Args:
      directory: DICOM dataset directory;
      name: file name.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the PNG image file.
    filename = os.path.join(directory, image_dir, name + '.jpg')
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    shape = list(f_jpeg_image_shape(image_data))

    # Get object annotations.
    labels = []
    labels_text = []
    bboxes = []

    # Read the txt label file, if it exists.
    filename2 = os.path.join(directory, label_dir, name + '.txt')
    if os.path.exists(filename2):
        with open(filename2) as f:
            label_data = f.readlines()
        for l in label_data:
            data = l.split()
            if len(data) > 0:
                # Label.
                labels.append(int(DICOM_LABELS[data[0]]))
                labels_text.append(data[0].encode('ascii'))
                # bbox.
#                bboxes.append((float(data[4]) / shape[1],
#                               float(data[5]) / shape[0],
#                               float(data[6]) / shape[1],
#                               float(data[7]) / shape[0]
#                               ))
                # bbox
                bboxes.append((0.0, 0.0, 1.0, 1.0)) # Set the entire image as bbox for classification task

    return (image_data, shape, labels, labels_text,  bboxes, filename)

def _convert_to_example(image_data, shape, labels, labels_text, bboxes, filename):
    """Build an Example proto for an image example.

    Args:
      image_data: string, PNG encoding of RGB image;
      labels: list of integers, identifier for the ground truth;
      labels_text: list of strings, human-readable labels;
      bboxes: list of bounding boxes; each box is a list of integers;
          specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong
          to the same label as the image label.
      shape: 3 integers, image shapes in pixels.
    Returns:
      Example proto
    """
    # Transpose bboxes, dimensions and locations.
    bboxes = list(map(list, zip(*bboxes)))

    # Iterators.
    it_bboxes = iter(bboxes)

    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data),
            'image/class/label': int64_feature(labels),
            'image/class/text': bytes_feature(labels_text),
            'image/object/bbox/xmin': float_feature(next(it_bboxes, [])),
            'image/object/bbox/ymin': float_feature(next(it_bboxes, [])),
            'image/object/bbox/xmax': float_feature(next(it_bboxes, [])),
            'image/object/bbox/ymax': float_feature(next(it_bboxes, [])),
            'image/object/class/label': int64_feature(labels),
            'image/filename': bytes_feature(filename.encode()),
            }))
    return example


def _add_to_tfrecord(dataset_dir, name, tfrecord_writer, f_jpeg_image_shape,
                     image_dir=DEFAULT_IMAGE_DIR, label_dir=DEFAULT_LABEL_DIR):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    l_data = _process_image(dataset_dir, name, f_jpeg_image_shape,
                            image_dir, label_dir)
    example = _convert_to_example(*l_data)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir, name):
    return '%s/%s.tfrecord' % (output_dir, name)


def run(dataset_dir, output_dir, name='dicom_train', shuffling=True, need_split='None'):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored
      output_dir: Output directory where tfrecord file will be stored
      name: Name of the output tfrecord
      need_split: (None): no split, (tv_split): train/validation split, (tvt_split): train/validation/test split
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    tf_filename = _get_output_filename(output_dir, name)
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # Dataset filenames, and shuffling.
    path = os.path.join(dataset_dir, DEFAULT_IMAGE_DIR)
    filenames = os.listdir(path)
    filenames.sort()

#    if shuffling:
#        random.seed(RANDOM_SEED)
#        random.shuffle(filenames)

    # jpeg decoding.
    inputs = tf.placeholder(dtype=tf.string)
    decoded_jpeg = tf.image.decode_jpeg(inputs, channels=3)
    with tf.Session() as sess:
        if need_split == 'None':
            # Process dataset files (No split)
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                for i, filename in enumerate(filenames):
                    sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                    sys.stdout.flush()

                    name = filename[:-4]
                    _add_to_tfrecord(dataset_dir, name, tfrecord_writer,
                                     lambda x: _jpeg_image_shape(x, sess, decoded_jpeg, inputs))
            print('\nFinished converting the DICOM dataset!')

        elif need_split == 'tt_split':
            # Process dataset files (train test split)
            train_filenames = filenames[NUM_TEST:]
            test_filenames = filenames[:NUM_TEST]

            with tf.python_io.TFRecordWriter(output_dir+'/train-dicom.tfrecord') as tfrecord_writer:
                for i, filename in enumerate(train_filenames):
                    sys.stdout.write('\r>> Converting image (Train) %d/%d' % (i+1, len(train_filenames)))
                    sys.stdout.flush()

                    name = filename[:-4]
                    _add_to_tfrecord(dataset_dir, name, tfrecord_writer,
                                     lambda x: _jpeg_image_shape(x, sess, decoded_jpeg, inputs))
            print('\nFinished converting the DICOM dataset (Train)!')

            with tf.python_io.TFRecordWriter(output_dir+'/test-dicom.tfrecord') as tfrecord_writer:
                for i, filename in enumerate(test_filenames):
                    sys.stdout.write('\r>> Converting image (Test) %d/%d' % (i+1, len(test_filenames)))
                    sys.stdout.flush()

                    name = filename[:-4]
                    _add_to_tfrecord(dataset_dir, name, tfrecord_writer,
                                     lambda x: _jpeg_image_shape(x, sess, decoded_jpeg, inputs))
            print('\nFinished converting the DICOM dataset (Test)!')

        elif need_split == 'tvt_split':
            # Process dataset files (train validation test split)
            train_filenames = filenames[NUM_TEST+NUM_VALIDATION:]
            validation_filenames = filenames[NUM_TEST:NUM_TEST+NUM_VALIDATION]
            test_filenames = filenames[:NUM_TEST]

            with tf.python_io.TFRecordWriter(output_dir+'/train-dicom.tfrecord') as tfrecord_writer:
                for i, filename in enumerate(train_filenames):
                    sys.stdout.write('\r>> Converting image (Train) %d/%d' % (i+1, len(train_filenames)))
                    sys.stdout.flush()

                    name = filename[:-4]
                    _add_to_tfrecord(dataset_dir, name, tfrecord_writer,
                                     lambda x: _jpeg_image_shape(x, sess, decoded_jpeg, inputs))
            print('\nFinished converting the DICOM dataset (Train)!')

            with tf.python_io.TFRecordWriter(output_dir+'/validation-dicom.tfrecord') as tfrecord_writer:
                for i, filename in enumerate(validation_filenames):
                    sys.stdout.write('\r>> Converting image (Validation) %d/%d' % (i+1, len(validation_filenames)))
                    sys.stdout.flush()

                    name = filename[:-4]
                    _add_to_tfrecord(dataset_dir, name, tfrecord_writer,
                                     lambda x: _jpeg_image_shape(x, sess, decoded_jpeg, inputs))
            print('\nFinished converting the DICOM dataset (Validation)!')

            with tf.python_io.TFRecordWriter(output_dir+'/test-dicom.tfrecord') as tfrecord_writer:
                for i, filename in enumerate(test_filenames):
                    sys.stdout.write('\r>> Converting image (Test) %d/%d' % (i+1, len(test_filenames)))
                    sys.stdout.flush()

                    name = filename[:-4]
                    _add_to_tfrecord(dataset_dir, name, tfrecord_writer,
                                     lambda x: _jpeg_image_shape(x, sess, decoded_jpeg, inputs))
            print('\nFinished converting the DICOM dataset (Test)!')
