# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
from src.vgg import Vgg16


class FcnModel(object):
    def __init__(self, input_tensor, y_true_tensor, is_training, n_classes=2):
        self._vgg = Vgg16(input_tensor, is_training)
        self._is_training = is_training
        self._n_classes = n_classes
        
        # public member : basic operations
        ################################################################
        self.inference_op = self._create_inference_op(n_classes)
        self.loss_op = self._create_loss_op(y_true_tensor)
        self.accuracy_op, self.iou_op, self.update_op = self._create_eval_op(y_true_tensor)
        self.summary_op = self._create_train_summary_op()
        ################################################################
        
    def _create_eval_op(self, y_true_tensor):
        logits = tf.reshape(self.inference_op, (-1, self._n_classes))
        probs = tf.nn.softmax(logits)
        pred_labels = tf.argmax(probs, 1)
        
        class_labels = tf.reshape(y_true_tensor, (-1, self._n_classes))
        labels = tf.argmax(class_labels, 1)
        is_correct = tf.equal(pred_labels, labels)
        accuracy_op = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        mean_iou_op, update_op = tf.metrics.mean_iou(labels, pred_labels, self._n_classes)
        return accuracy_op, mean_iou_op, update_op

    def _create_inference_op(self, num_classes):
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
        
        # inference_op : (H, W, num_classes)
        inference_op = _upsampling(layer3_decoder, num_classes, ratio=8)
        return inference_op

    def _create_loss_op(self, y_true_tensor):
        logits = tf.reshape(self.inference_op, (-1, self._n_classes))
        class_labels = tf.reshape(y_true_tensor, (-1, self._n_classes))
        
        # The cross_entropy_loss is the cost which we are trying to minimize to yield higher accuracy
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = class_labels)
        cross_entropy_loss = tf.reduce_mean(cross_entropy)
        return cross_entropy_loss

    def _create_train_summary_op(self):
        with tf.name_scope('train_summary'):
            summary_loss = tf.summary.scalar('loss', self.loss_op)
            summary_acc = tf.summary.scalar('pixelwise_accuracy', self.accuracy_op)
            
            # Todo: inference image를 summay에 추가
            summary_op = tf.summary.merge([summary_loss, summary_acc], name='train_summary')
            return summary_op


def _conv_1x1(input_layer, n_classes):
    return slim.conv2d(input_layer, n_classes, [1, 1], activation_fn=None)

def _upsampling(input_layer, n_classes, ratio=2):
    return slim.conv2d_transpose(input_layer,
                                 n_classes,
                                 [ratio*2, ratio*2],
                                 [ratio, ratio],
                                 activation_fn=None)
