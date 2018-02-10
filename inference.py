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

    x_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], 3])
    y_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    lr_placeholder = tf.placeholder(tf.float32)
    is_train_placeholder = tf.placeholder(tf.bool)

    fcn_model = FcnModel(x_placeholder, y_placeholder, is_train_placeholder, num_classes)
    get_batches_fn = gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, "models/model.ckpt")
        
        logits = tf.reshape(fcn_model.inference_op, (-1, num_classes))
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, is_train_placeholder, x_placeholder)
