# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
from src.fcn import FcnModel
from src.batch import gen_batch_function

DEFAULT_TEST_DIR = './data_tiny/data_road/training'
DEFAULT_MODEL_PATH = "models/model.ckpt"

argparser = argparse.ArgumentParser(description='Evaluate using pretrained model')
argparser.add_argument('-t',
                       '--test_dir',
                       default=DEFAULT_TEST_DIR,
                       help='path to dataset')
argparser.add_argument('-m',
                       '--model',
                       default=DEFAULT_MODEL_PATH,
                       help='path to saved model')


if __name__ == '__main__':
    num_classes = 2
    image_shape = (160, 576)
    args = argparser.parse_args()

    x_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], 3])
    y_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    lr_placeholder = tf.placeholder(tf.float32)
    is_train_placeholder = tf.placeholder(tf.bool)

    fcn_model = FcnModel(x_placeholder, y_placeholder, is_train_placeholder, num_classes)
    get_batches_fn = gen_batch_function(args.test_dir, image_shape)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, args.model)
        
        ious = 0
        n_samples = 0
        for images, labels in get_batches_fn(1):
            feed = {x_placeholder: images, y_placeholder: labels, is_train_placeholder : False}
            
            sess.run(tf.local_variables_initializer())
            sess.run(fcn_model.update_op, feed_dict = feed)
            iou = sess.run(fcn_model.iou_op, feed_dict = feed)
            segmentation_map = sess.run(tf.nn.softmax(fcn_model.inference_op),
                                        feed_dict = feed)
                                        
            import matplotlib.pyplot as plt
            images = [images[0], labels[0,:,:,1], segmentation_map[0,:,:,1]]
            titles = ["Original image",
                      "Ground truth road pixels",
                      "Inferenced pixels (iou-score: {:.2f})".format(iou)]
            
            fig, ax = plt.subplots(nrows=1, ncols=len(images))
            for i, (img, title) in enumerate(zip(images, titles)):
                plt.subplot(len(images), 1, i+1)
                plt.title(title, fontsize=15)
                plt.imshow(img)
                plt.axis('off')
            plt.tight_layout(h_pad=1.0)
            plt.savefig("{}.png".format(n_samples), bbox_inches='tight')
            print("{}.png".format(n_samples))
            n_samples += 1
            
#         mean_iou = ious/n_samples
#         print("mean-iou score: {:.3f}".format(mean_iou))
        
            
