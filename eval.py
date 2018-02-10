# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import helper

from src.utils import plot_img
from src.fcn import FcnModel
from src.batch import gen_batch_function

if __name__ == '__main__':
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data_tiny'
    runs_dir = './runs_tiny'
    test_path = os.path.join('./data_tiny', 'data_road/training')
    model_path = "models/model.ckpt"

    x_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], 3])
    y_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    lr_placeholder = tf.placeholder(tf.float32)
    is_train_placeholder = tf.placeholder(tf.bool)

    fcn_model = FcnModel(x_placeholder, y_placeholder, is_train_placeholder, num_classes)
    get_batches_fn = gen_batch_function(test_path, image_shape)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        
        ious = 0
        n_samples = 0
        for images, labels in get_batches_fn(1):
            feed = {x_placeholder: images, y_placeholder: labels, is_train_placeholder : False}
            
            sess.run(tf.local_variables_initializer())
            sess.run(fcn_model.update_op, feed_dict = feed)
            iou = sess.run(fcn_model.iou_op, feed_dict = feed)
            ious += iou
            n_samples += 1
            
        mean_iou = ious/n_samples
        print("mean-iou score: {:.3f}".format(mean_iou))
        
            
