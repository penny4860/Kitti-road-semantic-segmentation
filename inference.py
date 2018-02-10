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

import cv2
import numpy as np
import scipy.misc
def gen_test_output_video(sess, logits, is_train_pl, image_pl, video_file, image_shape):
    cap = cv2.VideoCapture(video_file)
    counter=0
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        image = scipy.misc.imresize(frame, image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {is_train_pl: False, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask_full = scipy.misc.imresize(mask, frame.shape)
        mask_full = scipy.misc.toimage(mask_full, mode="RGBA")
        mask = scipy.misc.toimage(mask, mode="RGBA")


        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        street_im_full = scipy.misc.toimage(frame)
        street_im_full.paste(mask_full, box=None, mask=mask_full)

        cv2.imwrite("video/{}.jpg".format(counter), np.array(street_im_full))
        print("{}.jpg".format(counter))
        counter=counter+1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

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

        gen_test_output_video(sess, logits, is_train_placeholder, x_placeholder, video_file, image_shape)
#         helper.save_inference_samples(args.runs,
#                                       args.dataset,
#                                       sess,
#                                       image_shape, 
#                                       logits,
#                                       is_train_placeholder,
#                                       x_placeholder)
