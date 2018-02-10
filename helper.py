import numpy as np
import os.path
import scipy.misc
import shutil
import time
import tensorflow as tf
from glob import glob


def gen_test_output(sess, logits, is_training, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {is_training: False, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)

import cv2
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


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, is_training, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, is_training, input_image, os.path.join(data_dir, 'data_road/training'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
