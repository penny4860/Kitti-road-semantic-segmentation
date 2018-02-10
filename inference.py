# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
import helper

from src.fcn import FcnModel

DATA_DIR = './data_tiny'
RUNS_DIR = './runs_tiny'

argparser = argparse.ArgumentParser(description='Inference using pretrained model')
argparser.add_argument('-d',
                       '--dataset',
                       default=DATA_DIR,
                       help='path to dataset')
argparser.add_argument('-r',
                       '--runs',
                       default=RUNS_DIR,
                       help='path to saved directory')

if __name__ == '__main__':
    image_shape = (160, 576)
    num_classes = 2

    x_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], 3])
    y_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    lr_placeholder = tf.placeholder(tf.float32)
    is_train_placeholder = tf.placeholder(tf.bool)
    fcn_model = FcnModel(x_placeholder, y_placeholder, is_train_placeholder, num_classes)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, "models/model.ckpt")
        
        logits = tf.reshape(fcn_model.inference_op, (-1, num_classes))
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, is_train_placeholder, x_placeholder)
