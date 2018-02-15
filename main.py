import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
from src.batch import gen_batch_function
import tensorflow.contrib.slim as slim

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg_ckpt(sess, ckpt='ckpts/vgg_16.ckpt'):
    variables = slim.get_variables(scope='vgg_16', suffix="weights") + slim.get_variables(scope='vgg_16', suffix="biases")
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(ckpt, variables)
    sess.run(init_assign_op, init_feed_dict)


def run():
    ##########################################
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    epochs = 25
    batch_size = 3
    ##########################################

    x_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], 3])
    y_placeholder = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    lr_placeholder = tf.placeholder(tf.float32)
    is_train_placeholder = tf.placeholder(tf.bool)
    
    from src.fcn import FcnModel
    fcn_model = FcnModel(x_placeholder, y_placeholder, is_train_placeholder, num_classes)
    train_op = tf.train.AdamOptimizer(lr_placeholder).minimize(fcn_model.loss_op)
    
    with tf.Session() as sess:
        # Create function to get batches
        get_batches_fn = gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
 
        # Todo:  Augment Images for better results
        sess.run(tf.global_variables_initializer())
        load_vgg_ckpt(sess, os.path.join(data_dir, 'vgg/vgg_16.ckpt'))

        ###############################################################################################
        for epoch in range(epochs):
            total_loss_value = 0
            for images, labels in get_batches_fn(batch_size):
                feed = {x_placeholder: images,
                        y_placeholder: labels,
                        lr_placeholder: 1e-2,
                        is_train_placeholder : True }
            
                _, loss_value = sess.run([train_op, fcn_model.loss_op], feed_dict = feed)
                total_loss_value += loss_value
                # print("loss : {:.2f}".format(loss_value))
            print("epoch: {}/{}, training loss: {:.2f}".format(epoch+1, epochs, total_loss_value))

#             feed = {x_placeholder: images,
#                     y_placeholder: labels,
#                     is_train_placeholder : False }
#             sess.run(tf.local_variables_initializer())
#             sess.run(fcn_model.update_op, feed_dict = feed)
#             loss_value, acc, iou = sess.run([fcn_model.loss_op, fcn_model.accuracy_op, fcn_model.iou_op], feed_dict = feed)
#             print("    loss: {:.3f}, accuracy: {:.3f}, iou: {:.3f}".format(loss_value, acc, iou))
        ###############################################################################################        
        
 
        saver = tf.train.Saver()
        saver.save(sess, "models/model.ckpt")
         
        # TODO: Save inference data using helper.save_inference_samples
        logits = tf.reshape(fcn_model.inference_op, (-1, num_classes))
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, is_train_placeholder, x_placeholder)
 
        # OPTIONAL: Apply the trained model to a video
        # Run the model with the test images and save each painted output image (roads painted green)


if __name__ == '__main__':
    run()
