# coding=utf-8
import os
from PIL import Image
import numpy as np
import tensorflow as tf

from models import model
from datasets.ocr_dataset import read_charset, DEFAULT_CONFIG
from train_OCR import create_mparams

FLAGS = tf.app.flags.FLAGS

"""使用ckpt文件评估模型"""


if __name__ == '__main__':
    path_img = './dataset_generate/data_sample/20455828_2605100732.jpg'
    path_ckpt = './train_logs/ocr_chinese_model.ckpt-1090000'

    charset = read_charset(os.path.join(FLAGS.path_dataset_root, DEFAULT_CONFIG['charset_filename']))

    ocr_model = model.Model(num_char_classes=len(charset),
                            seq_length=DEFAULT_CONFIG['max_sequence_length'],
                            num_views=DEFAULT_CONFIG['num_of_views'],
                            null_code=DEFAULT_CONFIG['null_code'],
                            mparams=create_mparams(),
                            charset=charset)
    shape_img = DEFAULT_CONFIG['image_shape']
    max_sequence_length = DEFAULT_CONFIG['max_sequence_length']
    pl_image = tf.placeholder(tf.float32, shape=[None, shape_img[0], shape_img[1], shape_img[2]], name='pl_image')
    endpoints = ocr_model.create_base(pl_image, labels_one_hot=None)
    init_fn = ocr_model.create_init_fn_to_restore(path_ckpt)

    resize_height = DEFAULT_CONFIG['image_shape'][0]
    resize_width = DEFAULT_CONFIG['image_shape'][1]
    img = Image.open(path_img)
    img = np.array(img.resize((resize_width, resize_height), Image.ANTIALIAS), dtype=np.float32)
    img = img[np.newaxis, :, :, :]

    with tf.Session() as sess:
        tf.tables_initializer().run()

        # sess.run(tf.global_variables_initializer()
        # step = path_ckpt.split('/')[-1].split('-')[-1]
        # print('eval model in {}'.format(path_ckpt))
        # saver = tf.train.import_meta_graph(path_ckpt + '.meta')
        # saver.restore(sess, path_ckpt)

        init_fn(sess)

        predictions = sess.run(endpoints.predicted_text, feed_dict={pl_image: img})
        for line in predictions:
            print(line)

