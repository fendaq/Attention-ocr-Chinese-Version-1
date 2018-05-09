# coding=utf-8
import os
import tensorflow as tf
from tensorflow.contrib import slim

import ocr_dataset


"""输入数据处理框架"""
FLAGS = tf.app.flags.FLAGS


def batch_input(split_name, batch_size, num_char_classes, dataset_dir=None, config=None):
    """
    数据集以 batch 的方式输入模型
    :return: 
    """
    if not config:
        config = ocr_dataset.DEFAULT_CONFIG
    if split_name not in config['splits']:
        raise ValueError('split name %s was not recognized.' % split_name)

    file_pattern = os.path.join(dataset_dir, config['splits'][split_name]['pattern'])
    files = tf.train.match_filenames_once(file_pattern)
    # 通过文件列表创建输入文件队列
    filename_queue = tf.train.string_input_producer(files, shuffle=False)

    # 设定的resize后的image的大小
    define_height = config['image_shape'][0]
    define_width = config['image_shape'][1]

    # 解析tfrecord文件
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/orig_width': tf.FixedLenFeature([], tf.int64),
        'image/class': tf.FixedLenFeature([config['max_sequence_length']], tf.int64),
        'image/unpadded_class': tf.VarLenFeature(tf.int64),
        'image/text': tf.FixedLenFeature([], tf.string)
    })
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [define_height, define_width, 3])
    label = features['image/class']
    unpaded_class = features['image/unpadded_class']
    # # one-hot 编码
    # label_one_hot = slim.one_hot_encoding(label, num_char_classes)

    # 将数据打包成batch
    min_after_dequeue = 30
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size, capacity, min_after_dequeue)

    return image_batch, label_batch, files

