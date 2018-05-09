# coding=utf-8
import tensorflow as tf
import numpy as np
from PIL import Image


if __name__ == '__main__':
    with tf.device(None):
        aa = tf.Variable(initial_value=[0], dtype=tf.float32)

    config = tf.ConfigProto()
    config.log_device_placement = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(aa)
