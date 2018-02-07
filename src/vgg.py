# -*- coding: utf-8 -*-

import tensorflow.contrib.slim as slim

class Vgg16(object):
    def __init__(self, input_tensor):
        self.input = input_tensor
        
        # Build convolutional layers only
        self.conv1_1 = slim.conv2d(input_tensor, 64, [3, 3], scope='vgg_16/conv1/conv1_1')
        self.conv1_2 = slim.conv2d(self.conv1_1, 64, [3, 3], scope='vgg_16/conv1/conv1_2')
        self.pool1 = slim.max_pool2d(self.conv1_2, [2, 2], scope='pool1')
        
        self.conv2_1 = slim.conv2d(self.pool1, 128, [3, 3], scope='vgg_16/conv2/conv2_1')
        self.conv2_2 = slim.conv2d(self.conv2_1, 128, [3, 3], scope='vgg_16/conv2/conv2_2')
        self.pool2 = slim.max_pool2d(self.conv2_2, [2, 2], scope='pool2')
        
        self.conv3_1 = slim.conv2d(self.pool2, 256, [3, 3], scope='vgg_16/conv3/conv3_1')
        self.conv3_2 = slim.conv2d(self.conv3_1, 256, [3, 3], scope='vgg_16/conv3/conv3_2')
        self.conv3_3 = slim.conv2d(self.conv3_2, 256, [3, 3], scope='vgg_16/conv3/conv3_3')
        self.pool3 = slim.max_pool2d(self.conv3_3, [2, 2], scope='pool3')
         
        self.conv4_1 = slim.conv2d(self.pool3, 512, [3, 3], scope='vgg_16/conv4/conv4_1')
        self.conv4_2 = slim.conv2d(self.conv4_1, 512, [3, 3], scope='vgg_16/conv4/conv4_2')
        self.conv4_3 = slim.conv2d(self.conv4_2, 512, [3, 3], scope='vgg_16/conv4/conv4_3')
        self.pool4 = slim.max_pool2d(self.conv4_3, [2, 2], scope='pool4')
         
        self.conv5_1 = slim.conv2d(self.pool4, 512, [3, 3], scope='vgg_16/conv5/conv5_1')
        self.conv5_2 = slim.conv2d(self.conv5_1, 512, [3, 3], scope='vgg_16/conv5/conv5_2')
        self.conv5_3 = slim.conv2d(self.conv5_2, 512, [3, 3], scope='vgg_16/conv5/conv5_3')
        self.pool5 = slim.max_pool2d(self.conv5_3, [2, 2], scope='pool5')
        
        # Use conv2d instead of fully_connected layers.
        self.pool6 = slim.conv2d(self.pool5, 4096, [7, 7], scope='vgg_16/fc6')
#         net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
#                          scope='dropout6')
        self.pool7 = slim.conv2d(self.pool6, 4096, [1, 1], scope='vgg_16/fc7')
        
#         self.layers = {'conv1_1' : self.conv1_1,
#                        'conv1_2' : self.conv1_2,
#                        'conv2_1' : self.conv2_1,
#                        'conv2_2' : self.conv2_2,
#                        'conv3_1' : self.conv3_1,
#                        'conv3_2' : self.conv3_2,
#                        'conv3_3' : self.conv3_3,
#                        'conv4_1' : self.conv4_1,
#                        'conv4_2' : self.conv4_2,
#                        'conv4_3' : self.conv4_3,
#                        'conv5_1' : self.conv5_1,
#                        'conv5_2' : self.conv5_2,
#                        'conv5_3' : self.conv5_3,
#                        
#                        }

    def load_ckpt(self, sess, ckpt='ckpts/vgg_16.ckpt'):
        variables = slim.get_variables(scope='vgg_16')
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(ckpt, variables)
        sess.run(init_assign_op, init_feed_dict)

#     def get_activation(self, layer_name='conv5_1'):
#         return self.layers[layer_name]

if __name__ == '__main__':
    import tensorflow as tf
    input_tensor = tf.placeholder(tf.float32, [None, None, None, 3])
    vgg = Vgg16(input_tensor)
    
    sess = tf.Session()
    vgg.load_ckpt(sess, ckpt="../data_tiny/vgg/vgg_16.ckpt")
    
    
    
    
    
    
    
    
    
