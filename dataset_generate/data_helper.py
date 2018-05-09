# coding=utf-8
import os
import codecs
import json
import glob
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt


"""制作数据集时的一些辅助函数"""


def size_train_val_test(path_root):
    """
    统计训练集，验证集，测试集大小
    :param path_root: 
    :return: 
    """
    with open(os.path.join(path_root, 'train.txt')) as f_train:
        contents = f_train.readlines()
        print('train set size: {}'.format(len(contents)))
    with open(os.path.join(path_root, 'val.txt')) as f_val:
        contents = f_val.readlines()
        print('val set size: {}'.format(len(contents)))
    with open(os.path.join(path_root, 'test.txt')) as f_test:
        contents = f_test.readlines()
        print('test set size: {}'.format(len(contents)))


def dataset_statistic(path_root, suffix):
    """
    统计数据集信息
    """
    abs_file_list = glob.glob(os.path.join(path_root, '*.' + suffix))
    print('{} {} files in {}'.format(len(abs_file_list), suffix, path_root))


def partition_train_val(path_txt, rate=0.2):
    """
    训练集验证集划分
    """
    with open(path_txt) as f_txt:
        contents = f_txt.readlines()
    shuffle_index = range(len(contents))
    random.shuffle(shuffle_index)
    len_val = int(len(contents) * rate)
    val = [contents[shuffle_index[idx]] for idx in range(len_val)]
    train = [contents[shuffle_index[idx]] for idx in range(len_val, len(contents))]

    # 保存为txt文件
    with open('val.txt', 'w') as f_val:
        need_write_content = ''
        for one_line in val:
            need_write_content += one_line
        f_val.writelines(need_write_content)
    with open('train.txt', 'w') as f_train:
        need_write_content = ''
        for one_line in train:
            need_write_content += one_line
        f_train.writelines(need_write_content)


def read_dict_txt(path_dict_txt):
    """
    读取中文字典的txt文件
    :return: 
    """
    chinese_dict = {}
    with codecs.open(path_dict_txt, encoding='utf-8') as dict_file:
        for line in dict_file:
            (key, value) = line.strip().split('\t')
            chinese_dict[int(key)] = value
    print('chinese dict is as follows:')
    print json.dumps(chinese_dict, ensure_ascii=False, encoding='UTF-8')

    return chinese_dict


def make_txt_annotation(one_line, root_dataset, chinese_dict, debug=False):
    """
    制作数据集中图片对应的 txt 真值文件
    :return: 
    """
    split_results = one_line.split(' ')
    filename_img = split_results[0]
    label = ''
    for idx in range(len(split_results) - 1):
        index = int(split_results[idx + 1])
        character = chinese_dict[index]
        label += character
    with codecs.open(os.path.join(root_dataset, 'images_txt_annotation', filename_img.replace('.jpg', '.txt')), 'w',
                     encoding='utf-8') as f_txt:
        f_txt.write(label)

    if debug:
        img = Image.open(os.path.join(root_dataset, 'images', filename_img))
        img = np.array(img)
        plt.figure()
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    dataset_root = '/home/user/cltdevelop/Code/dataset/OCR_Chinese_Dataset'
    size_train_val_test(dataset_root)

    # partition_train_val('/home/user/cltdevelop/Code/dataset/OCR_Chinese_Dataset/train_and_val.txt'

    # dataset_root = '/home/user/cltdevelop/Code/dataset/OCR_Chinese_Dataset'
    #
    # chinese_dict = read_dict_txt(os.path.join(dataset_root, 'new_dic.txt'))
    # with open(os.path.join(dataset_root, 'train_and_val.txt')) as f_txt:
    #     contents = f_txt.readlines()
    # for line in contents:
    #     print(line)
    #     make_txt_annotation(line, dataset_root, chinese_dict, False)
