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
"""Convert a dataset to TFRecords format, which can be easily integrated into
a TensorFlow pipeline.

Usage:
```shell
python tf_convert_data.py \
    --dataset_name=dicom \
    --dataset_dir=/data/sample/training \
    --output_name=dicom_train \
    --output_dir=/data/sample/training \
    --need_validation_split=False \ 
```
"""
import tensorflow as tf
from dataset import dicom_to_tfrecords

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name', 'dicom',
    'The name of the dataset to convert.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    'Directory where the original dataset is stored.')
tf.app.flags.DEFINE_string(
    'output_name', 'dicom',
    'Basename used for TFRecords output files.')
tf.app.flags.DEFINE_string(
    'output_dir', './',
    'Output directory where to store TFRecords files.')
tf.app.flags.DEFINE_string(
    'need_validation_split', 'False',
    'If true, splits the training dataset into train and validation set.')


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    print('Dataset directory:', FLAGS.dataset_dir)
    print('Output directory:', FLAGS.output_dir)

    if FLAGS.dataset_name == 'dicom':
        if FLAGS.need_validation_split == 'True':
            dicom_to_tfrecords.run(dataset_dir=FLAGS.dataset_dir, output_dir=FLAGS.output_dir, name=FLAGS.output_name, need_validation_split=True)
        else:
            dicom_to_tfrecords.run(dataset_dir=FLAGS.dataset_dir, output_dir=FLAGS.output_dir, name=FLAGS.output_name, need_validation_split=False)    
    else:
        raise ValueError('Dataset [%s] was not recognized.' % FLAGS.dataset_name)

if __name__ == '__main__':
    tf.app.run()