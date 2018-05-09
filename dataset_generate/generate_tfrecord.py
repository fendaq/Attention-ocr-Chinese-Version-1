# coding=utf-8
import os
import codecs
import json
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
"""原始数据集中相关路径设置"""
tf.app.flags.DEFINE_string('dir_dict_txt',
                           '/home/user/cltdevelop/Code/dataset/OCR_Chinese_Dataset/new_dic.txt',
                           'absolute path of chinese dict txt')
tf.app.flags.DEFINE_string('path_dataset_root',
                           '/home/user/cltdevelop/Code/dataset/OCR_Chinese_Dataset',
                           'path of data set root')
tf.app.flags.DEFINE_string('path_save_tfrecord_root',
                           '/home/user/cltdevelop/Code/dataset/OCR_Chinese_Dataset',
                           'where to save the generated tfrecord file')

"""原始数据集中的相关属性设置"""
tf.app.flags.DEFINE_string('suffix', 'jpg', 'suffix of image in data set')
tf.app.flags.DEFINE_string('height_and_width', '32, 240', 'input size of each image in model training')
tf.app.flags.DEFINE_integer('length_of_text', 20, 'length of text when this text is padded')
tf.app.flags.DEFINE_integer('null_char_id', 5462, 'the index of null char is used to padded text')


def read_chinese_dict():
    """
    读取中文字典的txt文件
    :return: 
    """
    chinese_dict = {}
    with codecs.open(FLAGS.dir_dict_txt, encoding='utf-8') as dict_file:
        for line in dict_file:
            (key, value) = line.strip().split('\t')
            chinese_dict[value] = int(key)
    print('chinese dict is as follows:')
    print json.dumps(chinese_dict, ensure_ascii=False, encoding='UTF-8')

    return chinese_dict


def _int64_feature(value):
    """
    返回整数型 Int64 feature
    :param value: 
    :return: 
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """
    返回字符串 bytes 类型的 feature
    :param value: 
    :return: 
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def encode_utf8_string(text, length, dic, null_char_id=5462):
    """
    对于每一个text, 返回对应的 pad 型和 unpaded 型的真值, 即在chinese dict中的索引
    :return: 
    """
    char_ids_padded = [null_char_id] * length
    char_ids_unpaded = [null_char_id] * len(text)
    for idx in range(len(text)):
        hash_id = dic[text[idx]]
        char_ids_padded[idx] = hash_id
        char_ids_unpaded[idx] = hash_id

    return char_ids_padded, char_ids_unpaded


def make_tfrecord(dict_chinese, dataset_name, num_tfrecord_files):
    """
    制作 tfrecord 文件
    :return: 
    """
    with open(os.path.join(FLAGS.path_dataset_root, dataset_name + '.txt')) as f_txt:
        contents = f_txt.readlines()
    contents = [each_line.split(' ')[0] for each_line in contents]
    # 所有图片对应的绝对路径列表
    addrs_image = [os.path.join(FLAGS.path_dataset_root, 'images', each_line) for each_line in contents]
    # 所有图片对应的真值txt文件列表
    addrs_label = [each.replace('.' + FLAGS.suffix, '.txt').replace('images', 'images_txt_annotation') for each
                   in addrs_image]
    print('{} images in {}'.format(len(addrs_image), FLAGS.path_dataset_root))
    # 图片resize的高和宽
    split_results = FLAGS.height_and_width.split(',')
    height = int(split_results[0].strip())
    width = int(split_results[1].strip())

    # 将数据平均分到 num_tfrecord_files 个tfrecords文件中来写入
    average_samples = len(addrs_image) // num_tfrecord_files
    samples_per_file = [average_samples] * (num_tfrecord_files - 1)
    samples_per_file.append(len(addrs_image) - average_samples * (num_tfrecord_files - 1))

    # 开始写入数据
    cnt = 0
    for file_i in range(num_tfrecord_files):
        filename = os.path.join(FLAGS.path_save_tfrecord_root, 'tfexample_' + dataset_name + '.tfrecords-%.5d-of-%.5d' %
                                (file_i, num_tfrecord_files))
        tfrecord_writer = tf.python_io.TFRecordWriter(filename)
        for i in range(samples_per_file[file_i]):
            path_img = addrs_image[cnt]
            path_label = addrs_label[cnt]
            print('{} / {}, {}'.format(i, samples_per_file[file_i], path_img))
            img = Image.open(path_img)
            orig_width = img.size[0]
            orig_height = img.size[1]
            img = img.resize((width, height), Image.ANTIALIAS)
            image_data = img.tobytes()
            with codecs.open(path_label, encoding="utf-8") as f_txt:
                text = f_txt.read()
            char_ids_padded, char_ids_unpadded = encode_utf8_string(text=text, length=FLAGS.length_of_text,
                                                                    dic=dict_chinese, null_char_id=FLAGS.null_char_id)
            one_sample = tf.train.Example(features=tf.train.Features(
                feature={
                    'image/encoded': _bytes_feature(image_data),
                    'image/format': _bytes_feature(b'raw'),
                    'image/width': _int64_feature([width]),
                    'image/orig_width': _int64_feature([orig_width]),
                    'image/class': _int64_feature(char_ids_padded),
                    'image/unpadded_class': _int64_feature(char_ids_unpadded),
                    'image/text': _bytes_feature(bytes(text.encode('utf-8')))
                }
            ))
            tfrecord_writer.write(one_sample.SerializeToString())
            cnt += 1
        tfrecord_writer.close()


def parse_tfrecord_file():
    """
    对生成的 tfrecords 文件进行解析，判断生成的 tfrecords 文件所包含的信息是否正确
    :return: 
    """
    # 创建一个reader来读取tfrecord文件中的样列
    reader = tf.TFRecordReader()
    # 创建一个队列来维护输入文件列表
    # filename_queue = tf.train.string_input_producer([FLAGS.path_save_tfrecord])
    # 注，files 是一个local variable，不会保存到checkpoint,需要用sess.run(tf.local_variables_initializer())初始化
    files = tf.train.match_filenames_once(FLAGS.path_save_tfrecord)
    filename_queue = tf.train.string_input_producer(files)
    # 读取一个样列
    _, serialized_example = reader.read(filename_queue)
    # 解析样列
    features = tf.parse_single_example(serialized_example, features={
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/orig_width': tf.FixedLenFeature([], tf.int64),
        'image/class': tf.FixedLenFeature([FLAGS.length_of_text], tf.int64),
        'image/unpadded_class': tf.VarLenFeature(tf.int64),
        'image/text': tf.FixedLenFeature([], tf.string)
    })

    # 设定的resize后的image的大小
    split_results = FLAGS.height_and_width.split(',')
    define_height = int(split_results[0].strip())
    define_width = int(split_results[1].strip())

    img = tf.decode_raw(features['image/encoded'], tf.uint8)
    img = tf.reshape(img, (define_height, define_width, 3))
    width = tf.cast(features['image/width'], tf.int32)
    ori_width = tf.cast(features['image/orig_width'], tf.int32)
    img_class = tf.cast(features['image/class'], tf.int32)
    img_unpaded_class = tf.cast(features['image/unpadded_class'], tf.int32)
    text = ''

    # 启动多线程处理输入数据
    coord = tf.train.Coordinator()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Starts all queue runners collected in the graph
        threads = tf.train.start_queue_runners(sess, coord)
        print(sess.run(files))
        for i in range(10):
            # 每次运行会自动读取tfrecord文件中的一个样列，当所有样列读取完后，会重头读取
            one_image, one_width, one_ori_width, one_img_class, one_img_unpaded_class = sess.run(
                [img, width, ori_width, img_class, img_unpaded_class])
            # 可视化解析出来的图片
            # one_image = np.reshape(one_image, (define_height, define_width, 3))
            plt.figure()
            plt.imshow(one_image)
            plt.show()


if __name__ == '__main__':
    chinese_dict = read_chinese_dict()
    make_tfrecord(chinese_dict, 'train', 10)
    make_tfrecord(chinese_dict, 'val', 3)

    # parse_tfrecord_file()

