# coding=utf-8
# coord = tf.train.Coordinator()
    # with tf.Session() as sess:
    #     # 启动线程
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.local_variables_initializer())
    #     thread = tf.train.start_queue_runners(sess, coord)
    #     # 查看一下数据
    #     for idx in range(10):
    #         batch_img_train, batch_label_train = sess.run([train_image_batch, train_label_batch])
    #         batch_one_hot = sess.run(train_one_hot)
    #         plt.figure()
    #         value_sqrt = int(math.ceil(math.sqrt(FLAGS.batch_size)))
    #         for j in range(FLAGS.batch_size):
    #             plt.subplot(value_sqrt, value_sqrt, j + 1)
    #             plt.imshow(batch_img_train[j, :, :, :])
    #         plt.show()