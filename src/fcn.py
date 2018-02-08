# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
from src.vgg import Vgg16


class FcnModel(object):
    def __init__(self, input_tensor, is_training):
        self._vgg = Vgg16(input_tensor, is_training)
        self._is_training = is_training
        
    def get_inference_op(self, num_classes):
        batch_norm_params = {'is_training': self._is_training,
                             'decay': 0.9,
                             'updates_collections': None}
    
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
        
            # layer7_decoder : (H/32, W/32, num_classes)
            layer7_decoder = _conv_1x1(self._vgg.pool7, num_classes)
        
            # layer4_decoder : (H/16, W/16, num_classes)
            layer4_decoder_in1 = _upsampling(layer7_decoder, num_classes, ratio=2)
            layer4_decoder_in2 = _conv_1x1(self._vgg.pool4, num_classes)
            layer4_decoder = tf.add(layer4_decoder_in1, layer4_decoder_in2)
        
            # layer3_decoder : (H/8, W/8, num_classes)
            layer3_decoder_in1 = _upsampling(layer4_decoder, num_classes, ratio=2)
            layer3_decoder_in2 = _conv_1x1(self._vgg.pool3, num_classes)
            layer3_decoder = tf.add(layer3_decoder_in1, layer3_decoder_in2)
        
        # decoder : (H, W, num_classes)
        inference_op = _upsampling(layer3_decoder, num_classes, ratio=8)
        return inference_op

    def load_vgg_ckpt(self, sess, ckpt='ckpts/vgg_16.ckpt'):
        variables = slim.get_variables(scope='vgg_16', suffix="weights") + slim.get_variables(scope='vgg_16', suffix="biases")
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(ckpt, variables)
        sess.run(init_assign_op, init_feed_dict)


def _conv_1x1(input_layer, n_classes):
    return slim.conv2d(input_layer, n_classes, [1, 1], activation_fn=None)

def _upsampling(input_layer, n_classes, ratio=2):
    return slim.conv2d_transpose(input_layer,
                                 n_classes,
                                 [ratio*2, ratio*2],
                                 [ratio, ratio],
                                 activation_fn=None)
