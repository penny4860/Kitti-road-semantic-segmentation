# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os

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
        saver = tf.train.Saver()
        saver.save(sess, "models/model.ckpt")
        saver.restore(sess, "models/model.ckpt")
        
        for images, labels in get_batches_fn(1):
            y_pred = sess.run(tf.nn.softmax(fcn_model.inference_op), feed_dict = {x_placeholder: images,
                                                                                  is_train_placeholder : False})
            sess.run(fcn_model.update_op, feed_dict = {x_placeholder: images,
                                                       y_placeholder: np.array(labels),
                                                       is_train_placeholder : False})
            accuracy, iou = sess.run([fcn_model.accuracy_op, fcn_model.iou_op], feed_dict = {x_placeholder: images,
                                                                                             y_placeholder: np.array(labels),
                                                                                             is_train_placeholder : False})
            print(accuracy, iou)
            plot_img([images[0], labels[0, :, :, 1], y_pred[0, :, :, 1]])
            break
            
            
            
