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


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, is_training):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    for epoch in range(epochs):
        total_loss_value = 0
        for images, labels in get_batches_fn(batch_size):
            feed = {input_image: images,
                    correct_label: labels,
                    keep_prob: 1.0,
                    learning_rate: 1e-1,
                    is_training : True }
        
            _, loss_value = sess.run([train_op, cross_entropy_loss], feed_dict = feed)
            total_loss_value += loss_value
            # print("loss : {:.2f}".format(loss_value))
        print("epoch: {}/{}, training loss: {:.2f}".format(epoch+1, epochs, total_loss_value))

def load_vgg_ckpt(sess, ckpt='ckpts/vgg_16.ckpt'):
    variables = slim.get_variables(scope='vgg_16', suffix="weights") + slim.get_variables(scope='vgg_16', suffix="biases")
    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(ckpt, variables)
    sess.run(init_assign_op, init_feed_dict)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data_tiny'
    runs_dir = './runs_tiny'
    # tests.test_for_kitti_dataset(data_dir)
    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    learning_rate = tf.placeholder(tf.float32)
    
    input_image = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], 3])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    
    from src.fcn import FcnModel
    fcn_model = FcnModel(input_image, correct_label, is_training, num_classes)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(fcn_model.loss_op)
    
    with tf.Session() as sess:
        # Create function to get batches
        get_batches_fn = gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
 
        # Todo:  Augment Images for better results
        sess.run(tf.global_variables_initializer())
        load_vgg_ckpt(sess, os.path.join(data_dir, 'vgg/vgg_16.ckpt'))
        
        train_nn(sess, 50, 2, get_batches_fn, 
                 train_op, fcn_model.loss_op, input_image,
                 correct_label, keep_prob, learning_rate, is_training)
 
        saver = tf.train.Saver()
        saver.save(sess, "models/model.ckpt")
        # saver.restore(sess, "models/model.ckpt")
         
        # TODO: Save inference data using helper.save_inference_samples
        logits = tf.reshape(fcn_model.inference_op, (-1, num_classes))
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, is_training, input_image)
 
        # OPTIONAL: Apply the trained model to a video
        # Run the model with the test images and save each painted output image (roads painted green)
        


if __name__ == '__main__':
    run()
