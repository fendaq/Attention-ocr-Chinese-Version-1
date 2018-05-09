# coding=utf-8
import os
import sys

import tensorflow as tf


"""将ckpt模型文件转化为pb文件"""


def obtain_node_name(path_ckpt):
    """
    获取训练模型中所有节点的名字
    :return: 
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(path_ckpt + '.meta')
        saver.restore(sess, path_ckpt)

        foo = [node.name for node in tf.get_default_graph().as_graph_def().node]
        for idx in range(len(foo)):
            print(foo[idx])

    # 将节点名字保存到txt文件
    with open('node_name.txt', 'w') as f_node:
        need_write_content = ''
        for one_node in foo:
            need_write_content += one_node + '\n'
        f_node.writelines(need_write_content)


def freeze_graph(path_ckpt, output_node_names):
    """
    将 ckpt 模型 freeze 成 pb 文件.
    """
    if not output_node_names:
        print('you need to supply the output node name')
        sys.exit()

    output_graph = 'frozen_ocr_chinese_model.pb'
    clear_device = True
    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(path_ckpt + '.meta', clear_devices=clear_device)
        saver.restore(sess, path_ckpt)
        # 将 variables 导出成 constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(),
                                                                        output_node_names.split(','))
        # 保存为 pb 文件
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print('%d ops in the final graph' % (len(output_graph_def.node)))

    return output_graph_def


if __name__ == '__main__':
    obtain_node_name('./train_logs/ocr_chinese_model.ckpt-185000')

    # path_ckpt = './train_logs/ocr_chinese_model.ckpt-185000'
    # output_node_names = ''
    # output_node_names += 'AttentionOcr_v1/chars_log_prob_tmp,'
    # output_node_names += 'AttentionOcr_v1/predicted_chars_tmp,'
    # output_node_names += 'AttentionOcr_v1/predicted_scores_tmp,'
    # output_node_names += 'AttentionOcr_v1/predicted_text_tmp'
    # freeze_graph(path_ckpt, output_node_names)
