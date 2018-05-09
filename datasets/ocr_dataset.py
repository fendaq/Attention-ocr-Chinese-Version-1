# coding=utf-8
import os
import re
import sys
import json
import glob

import tensorflow as tf
from tensorflow.contrib import slim


DEFAULT_CONFIG = {
    'name': 'OCR_DATASET',
    'splits': {
        'train': {
            'size': 2623685,
            'pattern': 'tfexample_train*'
        },
        'test': {
            'size': 364400,
            'pattern': 'tfexample_test*'
        },
        'val': {
            'size': 655921,
            'pattern': 'tfexample_val*'
        }
    },
    'charset_filename':
        'new_dic.txt',
    'image_shape': (32, 240, 3),
    'num_of_views':
        1,
    'max_sequence_length':
        20,
    'null_code':
        5462,
    'items_to_descriptions': {
        'image':
            'A [32 x 240 x 3] color image.',
        'label':
            'Characters codes.',
        'text':
            'A unicode string.',
        'length':
            'A length of the encoded text.',
        'num_of_views':
            'A number of different views stored within the image.'
    }
}


def read_charset(filename, null_character=u'\u2591'):
    """
    Reads a charset definition from a tab separated text file
    :param filename: 
    :param null_character: a unicode character used to replace '<null>' character. the
        default value is a light shade block '░'.
    :return: 
    """
    pattern = re.compile(r'(\d+)\t(.+)')
    charset = {}
    with tf.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                print('incorrect charset file. line #%d: %s', i, line)
                continue
            code = int(m.group(1))
            char = m.group(2)
            if char == '<nul>':
                char = null_character
            charset[code] = char

    return charset


class _NumOfViewsHandler(slim.tfexample_decoder.ItemHandler):
    """Convenience handler to determine number of views stored in an image."""

    def __init__(self, width_key, original_width_key, num_of_views):
        super(_NumOfViewsHandler, self).__init__([width_key, original_width_key])
        self._width_key = width_key
        self._original_width_key = original_width_key
        self._num_of_views = num_of_views

    def tensors_to_item(self, keys_to_tensors):
        return tf.to_int64(self._num_of_views * keys_to_tensors[self._original_width_key] /
                           keys_to_tensors[self._width_key])


def get_split(split_name, dataset_dir=None, config=None):
    """
    根据 split_name 获取相应的数据集, split_name的可选值：
    :return: Returns a dataset tuple for ocr dataset
    """
    if dataset_dir is None:
        print('please specify the path of dataset')
        sys.exit()
    if not config:
        config = DEFAULT_CONFIG
    if split_name not in config['splits']:
        raise ValueError('split name %s was not recognized.' % split_name)

    # 忽略 tfrecord 中的'image/height' 特征
    zero = tf.zeros([1], dtype=tf.int64)
    # 和tfrecord中<string, value>对应
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/format': tf.FixedLenFeature([], tf.string, default_value='jpg'),
        'image/width': tf.FixedLenFeature([1], tf.int64, default_value=zero),
        'image/orig_width': tf.FixedLenFeature([1], tf.int64, default_value=zero),
        'image/class': tf.FixedLenFeature([config['max_sequence_length']], tf.int64),
        'image/unpadded_class': tf.VarLenFeature(tf.int64),
        'image/text': tf.FixedLenFeature([1], tf.string, default_value='')
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=config['image_shape'], image_key='image/encoded',
                                              format_key='image/format'),
        'label': slim.tfexample_decoder.Tensor(tensor_key='image/class'),
        'text': slim.tfexample_decoder.Tensor(tensor_key='image/text'),
        'num_of_views': _NumOfViewsHandler(width_key='image/width', original_width_key='image/orig_width',
                                           num_of_views=config['num_of_views'])
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    charset_file = os.path.join(dataset_dir, config['charset_filename'])
    charset = read_charset(charset_file)
    print('chinese dict is as follows:')
    print(json.dumps(charset, ensure_ascii=False, encoding='UTF-8'))

    file_pattern = os.path.join(dataset_dir, config['splits'][split_name]['pattern'])

    # 返回的是包含各属性封装好的元组
    return slim.dataset.Dataset(data_sources=file_pattern,
                                reader=tf.TFRecordReader,
                                decoder=decoder,
                                num_samples=config['splits'][split_name]['size'],
                                items_to_descriptions=config['items_to_descriptions'],
                                # 额外的其他参数
                                charset=charset,
                                num_char_classes=len(charset),
                                num_of_views=config['num_of_views'],
                                max_sequence_length=config['max_sequence_length'],
                                null_code=config['null_code'])


if __name__ == '__main__':
    dataset = get_split('train', '../dataset_generate', None)
    print dataset.num_char_classes



