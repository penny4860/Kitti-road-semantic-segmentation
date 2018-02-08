# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim

def fcn(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, is_training):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    batch_norm_params = {'is_training': is_training,
                         'decay': 0.9,
                         'updates_collections': None}

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):

#         self.conv1_1 = slim.conv2d(rgb, 64, [3, 3], scope='vgg_16/conv1/conv1_1')
#         self.conv1_2 = slim.conv2d(self.conv1_1, 64, [3, 3], scope='vgg_16/conv1/conv1_2')
#         self.pool1 = slim.max_pool2d(self.conv1_2, [2, 2], scope='pool1')
    
        # layer7_decoder : (H/32, W/32, num_classes)
        layer7_decoder = conv_1x1(vgg_layer7_out, num_classes)
    
        # layer4_decoder : (H/16, W/16, num_classes)
        layer4_decoder_in1 = upsampling(layer7_decoder, num_classes, ratio=2)
        layer4_decoder_in2 = conv_1x1(vgg_layer4_out, num_classes)
        layer4_decoder = tf.add(layer4_decoder_in1, layer4_decoder_in2)
    
        # layer3_decoder : (H/8, W/8, num_classes)
        layer3_decoder_in1 = upsampling(layer4_decoder, num_classes, ratio=2)
        layer3_decoder_in2 = conv_1x1(vgg_layer3_out, num_classes)
        layer3_decoder = tf.add(layer3_decoder_in1, layer3_decoder_in2)
    
    # decoder : (H, W, num_classes)
    decoder = upsampling(layer3_decoder, num_classes, ratio=8)
    return decoder

def conv_1x1(input_layer, n_classes):
    return slim.conv2d(input_layer, n_classes, [1, 1], activation_fn=None)
#     return tf.layers.conv2d(input_layer, n_classes, 1,
#                             strides=(1,1),
#                             padding= 'same',
#                             kernel_initializer= tf.random_normal_initializer(stddev=0.01),
#                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

def upsampling(input_layer, n_classes, ratio=2):
    return slim.conv2d_transpose(input_layer,
                                 n_classes,
                                 [ratio*2, ratio*2],
                                 [ratio, ratio],
                                 activation_fn=None)
    
    # input_layer, n_classes, [1, 1], activation_fn=None
#     return tf.layers.conv2d_transpose(input_layer, n_classes, ratio*2,
#                                       strides=(ratio, ratio),
#                                       padding= 'same', 
#                                       kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
#                                       kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))



