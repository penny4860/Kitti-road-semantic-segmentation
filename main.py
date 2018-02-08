import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from src.batch import gen_batch_function


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Reshape 4D tensors to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    class_labels = tf.reshape(correct_label, (-1, num_classes))
    
    # The cross_entropy_loss is the cost which we are trying to minimize to yield higher accuracy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = class_labels)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)
    
    # The model implements this operation to find the weights/parameters that would yield correct pixel labels
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


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

# tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data_tiny'
    runs_dir = './runs_tiny'
    # tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    learning_rate = tf.placeholder(tf.float32)
    
    input_image = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], 3])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    
    from src.vgg import Vgg16
    from src.fcn import fcn
    vgg16 = Vgg16(input_image, is_training)
    model_output = fcn(vgg16.pool3, vgg16.pool4, vgg16.pool7, num_classes, is_training)
    logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, num_classes)
    
    with tf.Session() as sess:
        # Create function to get batches
        get_batches_fn = gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
 
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        # TODO: Build NN using load_vgg, layers, and optimize function
        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        vgg16.load_ckpt(sess, 'data_tiny/vgg/vgg_16.ckpt')
 
        train_nn(sess, 50, 2, get_batches_fn, 
                 train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate, is_training)
 
        saver = tf.train.Saver()
        saver.save(sess, "models/model.ckpt")
        # saver.restore(sess, "models/model.ckpt")
         
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, is_training, input_image)
 
        # OPTIONAL: Apply the trained model to a video
        # Run the model with the test images and save each painted output image (roads painted green)
        


if __name__ == '__main__':
    run()
