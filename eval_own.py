# coding=utf-8
import os
from PIL import Image
import numpy as np
from datasets.ocr_dataset import DEFAULT_CONFIG

import tensorflow as tf


"""模型测试"""


def load_graph(path_pb):
    """
    从 pb 文件中加载图模型
    :param path_pb: 
    :return: 
    """
    with tf.gfile.GFile(path_pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    return graph


def model_test(path_pb, path_img):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    graph = load_graph(path_pb)
    # 验证一下，能否访问到原始图中的operation
    for op in graph.get_operations():
        print(op.name)

    # 访问输入和输出节点
    input_images = graph.get_tensor_by_name('import/pl_image:0')
    chars_log_prob = graph.get_tensor_by_name('import/AttentionOcr_v1/chars_log_prob_tmp:0')
    predicted_chars = graph.get_tensor_by_name('import/AttentionOcr_v1/predicted_chars_tmp:0')
    predicted_scores = graph.get_tensor_by_name('import/AttentionOcr_v1/predicted_scores_tmp:0')
    predicted_text = graph.get_tensor_by_name('import/AttentionOcr_v1/predicted_text_tmp:0')
    # 开启会话，进行测试
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=graph) as sess:
        resize_height = DEFAULT_CONFIG['image_shape'][0]
        resize_width = DEFAULT_CONFIG['image_shape'][1]
        img = Image.open(path_img)
        img = np.array(img.resize((resize_width, resize_height), Image.ANTIALIAS))
        img = img[np.newaxis, :, :, :]
        predictions = sess.run(predicted_text, feed_dict={input_images: img})
        results = predictions.tolist()
        print results


if __name__ == '__main__':
    path_pb = 'frozen_ocr_chinese_model.pb'
    path_img = './dataset_generate/data_sample/20455828_2605100732.jpg'
    model_test(path_pb, path_img)
