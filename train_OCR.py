# coding=utf-8
import os
import sys
import math
import json
import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim

from datasets.ocr_dataset import DEFAULT_CONFIG, read_charset
from datasets.image_input_pipeline import batch_input
from models import model


FLAGS = tf.app.flags.FLAGS
"""模型训练的基本配置"""
tf.app.flags.DEFINE_string('path_dataset_root',
                           '/home/user/cltdevelop/Code/dataset/OCR_Chinese_Dataset',
                           'absolute root path of data set which contains tfrecords files and new_dic.txt file')
tf.app.flags.DEFINE_string('split_name', 'train', 'choose dataset type: train or test or val')
tf.app.flags.DEFINE_integer('train_batch_size', 128, 'batch size of training set.')
tf.app.flags.DEFINE_integer('val_batch_size', 128, 'batch size of validation set')
tf.app.flags.DEFINE_integer('crop_width', None, 'Width of the central crop for images.')
tf.app.flags.DEFINE_integer('crop_height', None, 'Height of the central crop for images.')
tf.app.flags.DEFINE_integer('num_epochs', 1000, 'number of epochs for training')
tf.app.flags.DEFINE_bool('use_augment_input', True, 'If True will use image augmentation')

"""模型结构的重要参数"""
tf.app.flags.DEFINE_string('final_endpoint', 'Mixed_5d', 'Endpoint to cut inception tower')
tf.app.flags.DEFINE_bool('use_attention', True, 'If True will use the attention mechanism')
tf.app.flags.DEFINE_bool('use_autoregression', True, 'If True will use autoregression (a feedback link)')
tf.app.flags.DEFINE_integer('num_lstm_units', 256, 'number of LSTM hidden units for sequence LSTM')

"""优化方法与正则"""
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'the optimizer to use')  # momentum
tf.app.flags.DEFINE_string('momentum', 0.9, 'momentum value for the momentum optimizer if used')
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'weight decay for char prediction FC layers')
tf.app.flags.DEFINE_float('lstm_state_clip_value', 10.0, 'cell state is clipped by this value prior to the cell '
                                                         'output activation')
# slim.learning
tf.app.flags.DEFINE_float('clip_gradient_norm', 2.0, 'If greater than 0 then the gradients would be clipped by it.')
# sequence_loss_fn
tf.app.flags.DEFINE_float('label_smoothing', 0.1, 'weight for label smoothing')
tf.app.flags.DEFINE_bool('ignore_nulls', True, 'ignore null characters for computing the loss')
tf.app.flags.DEFINE_bool('average_across_timesteps', False, 'divide the returned cost by the total label weight')

"""模型的存储与显示"""
tf.app.flags.DEFINE_string('train_log_dir', './train_logs', 'Directory where to write event logs.')
tf.app.flags.DEFINE_integer('save_step', 2000, 'save model when it is trained save_step each.')
tf.app.flags.DEFINE_integer('display_step', 100, 'display training information with each display_step steps')

"""中断模型继续训练或者使用已有模型finetune"""
tf.app.flags.DEFINE_bool('resume', True, 'whether to resume model from broken point')
tf.app.flags.DEFINE_bool('fine_tune', False, 'whether to fine tune model')             # finetune 不成功
tf.app.flags.DEFINE_string('checkpoint_inception', './pretrained_models/inception_v3.ckpt',
                           'Checkpoint to recover inception weights from.')
tf.app.flags.DEFINE_string('checkpoint_all', './train_logs/ocr_chinese_model.ckpt-1090000',
                           'Path for checkpoint to restore weights from.')


def prepare_training_dir():
    """
    数据集及文件夹路径检查
    :return: 
    """
    if not tf.gfile.Exists(FLAGS.train_log_dir):
        tf.gfile.MakeDirs(FLAGS.train_log_dir)


def create_mparams():
    """
    设定ocr序列模型中用到的一些参数
    :return: 
    """
    return {
        'conv_tower_fn': model.ConvTowerParams(final_endpoint=FLAGS.final_endpoint),
        'sequence_logit_fn': model.SequenceLogitsParams(
            use_attention=FLAGS.use_attention,
            use_autoregression=FLAGS.use_autoregression,
            num_lstm_units=FLAGS.num_lstm_units,
            weight_decay=FLAGS.weight_decay,
            lstm_state_clip_value=FLAGS.lstm_state_clip_value),
        'sequence_loss_fn': model.SequenceLossParams(
            label_smoothing=FLAGS.label_smoothing,
            ignore_nulls=FLAGS.ignore_nulls,
            average_across_timesteps=FLAGS.average_across_timesteps)
    }


def get_crop_size():
    if FLAGS.crop_width and FLAGS.crop_height:
        return (FLAGS.crop_width, FLAGS.crop_height)
    else:
        return None


def _init_weight(sess):
    """
    网络权重参数初始化
    :param sess: 
    :return: 
    """
    if FLAGS.resume and FLAGS.fine_tune:
        raise Exception("There should be only one mode")

    if FLAGS.resume:
        # 从断点处继续训练
        step = FLAGS.checkpoint_all.split('/')[-1].split('-')[-1]
        print("resume training from step {}".format(step))
        saver = tf.train.import_meta_graph(FLAGS.checkpoint_all + '.meta')
        saver.restore(sess, FLAGS.checkpoint_all)
        print('continue training from step: {}'.format(step))

    elif FLAGS.fine_tune:
        sess.run(tf.global_variables_initializer())
        scope = 'AttentionOcr_v1/conv_tower_fn/INCE'
        # 需要排除的变量
        exclusions = []
        exclusions.append('Momentum')

        variable_map = {}
        method_variables = slim.get_variables_to_restore(include=[scope])
        for var in method_variables:
            excluded = False
            for exclusion in exclusions:
                if exclusion in var.op.name:
                    excluded = True
                    break
            if not excluded:
                # 去掉前面的prefix
                var_name = var.op.name[len(scope) + 1:]
                variable_map[var_name] = var
        saver = tf.train.Saver(variable_map)
        saver.restore(sess, FLAGS.checkpoint_inception)
        print('finish load finetuned model')
        step = 0
    else:
        print('train model from scratch')
        init = tf.global_variables_initializer()
        sess.run(init)
        step = 0

    return int(step)


def create_optimizer():
    """
    优化器选择及参数配置
    :return: 
    """
    if FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, momentum=FLAGS.momentum)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    elif FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate)
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum)
    else:
        print('wrong optimizer specified in create_optimizer()')
        sys.exit()
    return optimizer


def main(_):
    # 设置模型训练时输出信息等级
    tf.logging.set_verbosity(tf.logging.INFO)

    ########################
    # 文件夹及数据集路径检查   #
    ########################
    prepare_training_dir()

    ########################
    # 数据集准备             #
    ########################
    charset = read_charset(os.path.join(FLAGS.path_dataset_root, DEFAULT_CONFIG['charset_filename']))
    print('chinese dict is as follows:')
    print(json.dumps(charset, ensure_ascii=False, encoding='UTF-8'))

    train_image_batch, train_label_batch, tfrecord_files = batch_input('train', FLAGS.train_batch_size, len(charset),
                                                                       FLAGS.path_dataset_root, None)
    train_one_hot = slim.one_hot_encoding(train_label_batch, len(charset))
    # val_image_batch, val_label_batch = batch_input('val', FLAGS.val_batch_size, len(charset),
    #                                                FLAGS.path_dataset_root, None)

    ########################
    # 模型构建             #
    ########################
    shape_img = DEFAULT_CONFIG['image_shape']
    max_sequence_length = DEFAULT_CONFIG['max_sequence_length']
    pl_image = tf.placeholder(tf.float32, shape=[None, shape_img[0], shape_img[1], shape_img[2]], name='pl_image')
    pl_label = tf.placeholder(tf.int64, shape=[None, max_sequence_length], name='pl_label')
    one_hot_label = slim.one_hot_encoding(pl_label, len(charset))
    ocr_model = model.Model(num_char_classes=len(charset),
                            seq_length=DEFAULT_CONFIG['max_sequence_length'],
                            num_views=DEFAULT_CONFIG['num_of_views'],
                            null_code=DEFAULT_CONFIG['null_code'],
                            mparams=create_mparams())
    endpoints = ocr_model.create_base(pl_image, one_hot_label)
    chars_logit = endpoints.chars_logit
    predicted_text = endpoints.predicted_text
    total_loss = ocr_model.create_loss_v2(pl_label, endpoints)

    ########################
    # 优化器配置            #
    ########################
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False,
                                  dtype=tf.int32)
    optimizer = create_optimizer()
    grads = optimizer.compute_gradients(total_loss)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)

    ########################
    # 配置并开始训练          #
    ########################
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # 权重参数初始化
    sess.run(tf.local_variables_initializer())
    step = _init_weight(sess)
    print('tfrecord files for training: {}'.format(sess.run(tfrecord_files)))
    # 线程协调器
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 训练集样本总数
    num_per_epoch = DEFAULT_CONFIG['splits']['train']['size']
    num_per_step = FLAGS.train_batch_size
    epoch = step * num_per_step // num_per_epoch
    while epoch < FLAGS.num_epochs:
        batch_img_train, batch_label_train = sess.run([train_image_batch, train_label_batch])
        # 模型在验证集上的评估
        ###########

        # 模型训练信息显示
        if step % FLAGS.display_step == 0:
            _ = sess.run(train_op, feed_dict={pl_image: batch_img_train, pl_label: batch_label_train})
            loss_train = sess.run(total_loss, feed_dict={pl_image: batch_img_train, pl_label: batch_label_train})
            print('epoch: {}, step: {}, train_loss: {}'.format(epoch, step, loss_train))
        else:
            _ = sess.run(train_op, feed_dict={pl_image: batch_img_train, pl_label: batch_label_train})
            # aa = sess.run(predicted_text, feed_dict={pl_image: batch_img_train, pl_label: batch_label_train})
            # bb = sess.run(chars_logit, feed_dict={pl_image: batch_img_train, pl_label: batch_label_train})

        # 模型保存
        if step % FLAGS.save_step == 0:
            checkpoint_path = os.path.join(FLAGS.train_log_dir, 'ocr_chinese_model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            print('************save model at {} steps'.format(step))

        step += 1
        epoch = step * num_per_step // num_per_epoch

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    tf.app.run()
