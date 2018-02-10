# -*- coding: utf-8 -*-
import argparse
import tensorflow as tf
import helper

from src.fcn import FcnModel

DEFAULT_DATA_DIR = './data_tiny'
DEFAULT_RUNS_DIR = './runs_tiny'
DEFAULT_MODEL_PATH = "models/model.ckpt"

argparser = argparse.ArgumentParser(description='Inference using pretrained model')
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
                       help='path to saved model')


if __name__ == '__main__':
    
    video_file = "video/test_video.mp4"
    
    image_shape = (160, 576)
    num_classes = 2
    args = argparser.parse_args()

    x_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], 3])
    y_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    lr_placeholder = tf.placeholder(tf.float32)
    is_train_placeholder = tf.placeholder(tf.bool)
    fcn_model = FcnModel(x_placeholder, y_placeholder, is_train_placeholder, num_classes)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, args.model)
        
        logits = tf.reshape(fcn_model.inference_op, (-1, num_classes))
        
        helper.gen_test_output_video(sess, logits, is_train_placeholder, x_placeholder, video_file, image_shape)
#         helper.save_inference_samples(args.runs,
#                                       args.dataset,
#                                       sess,
#                                       image_shape, 
#                                       logits,
#                                       is_train_placeholder,
#                                       x_placeholder)
