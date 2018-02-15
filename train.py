# -*- coding: utf-8 -*-
import argparse
import os.path
import tensorflow as tf
import numpy as np
import helper
from src.batch import gen_batch_function
import tensorflow.contrib.slim as slim

from src.fcn import FcnModel

DEFAULT_DATA_DIR = './data_tiny'
DEFAULT_RUNS_DIR = './runs_tiny'
DEFAULT_MODEL_PATH = "models/model.ckpt"
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 2

argparser = argparse.ArgumentParser(description='Training')
argparser.add_argument('-d',
                       '--dataset',
                       default=DEFAULT_DATA_DIR,
                       help='path to dataset')
argparser.add_argument('-r',
                       '--runs',
                       default=DEFAULT_RUNS_DIR,
                       help='path to saved directory')
argparser.add_argument('-m',
                       '--model',
                       default=DEFAULT_MODEL_PATH,
                       help='path to save model')
argparser.add_argument('-e',
                       '--epochs',
                       default=DEFAULT_EPOCHS,
                       help='number of epochs')
argparser.add_argument('-b',
                       '--batch',
                       default=DEFAULT_BATCH_SIZE,
                       help='batch size')


def load_vgg_ckpt(sess, ckpt='ckpts/vgg_16.ckpt'):
    variables = slim.get_variables(scope='vgg_16', suffix="weights") + slim.get_variables(scope='vgg_16', suffix="biases")
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(ckpt, variables)
    sess.run(init_assign_op, init_feed_dict)


def run():
    image_shape = (160, 576)
    num_classes = 2
    args = argparser.parse_args()

    x_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], 3])
    y_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    lr_placeholder = tf.placeholder(tf.float32)
    is_train_placeholder = tf.placeholder(tf.bool)
    fcn_model = FcnModel(x_placeholder, y_placeholder, is_train_placeholder, num_classes)

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    train_op = tf.train.AdamOptimizer(lr_placeholder).minimize(fcn_model.loss_op, global_step)
    
    with tf.Session() as sess:
        # Create function to get batches
        get_batches_fn = gen_batch_function(os.path.join(args.dataset, 'data_road/training'),
                                            image_shape)
 
        # Todo:  Augment Images for better results
        sess.run(tf.global_variables_initializer())
        load_vgg_ckpt(sess, os.path.join(args.dataset, 'vgg/vgg_16.ckpt'))
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('graphs/train', sess.graph)

        ###############################################################################################
        best_loss = np.inf
        for epoch in range(int(args.epochs)):
            total_loss_value = 0
            for images, labels in get_batches_fn(int(args.batch)):
                feed = {x_placeholder: images,
                        y_placeholder: labels,
                        lr_placeholder: 1e-2,
                        is_train_placeholder : True }
            
                _, loss_value, summary_value = sess.run([train_op, fcn_model.loss_op, fcn_model.summary_op],
                                                        feed_dict = feed)
                total_loss_value += loss_value
                
                writer.add_summary(summary_value, sess.run(global_step))
                # print("loss : {:.2f}".format(loss_value))
            print("epoch: {}/{}, training loss: {:.2f}".format(epoch+1, int(args.epochs), total_loss_value))
            if total_loss_value < best_loss:
                saver.save(sess, "models/model.ckpt")
                print("    best model update!!!")
                        
            
#             feed = {x_placeholder: images,
#                     y_placeholder: labels,
#                     is_train_placeholder : False }
#             sess.run(tf.local_variables_initializer())
#             sess.run(fcn_model.update_op, feed_dict = feed)
#             loss_value, acc, iou = sess.run([fcn_model.loss_op, fcn_model.accuracy_op, fcn_model.iou_op], feed_dict = feed)
#             print("    loss: {:.3f}, accuracy: {:.3f}, iou: {:.3f}".format(loss_value, acc, iou))
        ###############################################################################################        
        
        # TODO: Save inference data using helper.save_inference_samples
        logits = tf.reshape(fcn_model.inference_op, (-1, num_classes))
        helper.save_inference_samples(args.runs, args.dataset, sess, image_shape, logits, is_train_placeholder, x_placeholder)
 
        # OPTIONAL: Apply the trained model to a video
        # Run the model with the test images and save each painted output image (roads painted green)


if __name__ == '__main__':
    run()
