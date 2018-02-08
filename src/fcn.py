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
    
        # layer7_decoder : (H/32, W/32, num_classes)
        layer7_decoder = _conv_1x1(vgg_layer7_out, num_classes)
    
        # layer4_decoder : (H/16, W/16, num_classes)
        layer4_decoder_in1 = _upsampling(layer7_decoder, num_classes, ratio=2)
        layer4_decoder_in2 = _conv_1x1(vgg_layer4_out, num_classes)
        layer4_decoder = tf.add(layer4_decoder_in1, layer4_decoder_in2)
    
        # layer3_decoder : (H/8, W/8, num_classes)
        layer3_decoder_in1 = _upsampling(layer4_decoder, num_classes, ratio=2)
        layer3_decoder_in2 = _conv_1x1(vgg_layer3_out, num_classes)
        layer3_decoder = tf.add(layer3_decoder_in1, layer3_decoder_in2)
    
    # decoder : (H, W, num_classes)
    decoder = _upsampling(layer3_decoder, num_classes, ratio=8)
    return decoder

def _conv_1x1(input_layer, n_classes):
    return slim.conv2d(input_layer, n_classes, [1, 1], activation_fn=None)

def _upsampling(input_layer, n_classes, ratio=2):
    return slim.conv2d_transpose(input_layer,
                                 n_classes,
                                 [ratio*2, ratio*2],
                                 [ratio, ratio],
                                 activation_fn=None)
