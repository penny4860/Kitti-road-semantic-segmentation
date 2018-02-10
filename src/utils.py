
import matplotlib.pyplot as plt

def plot_img(images, show=True, save_filename=None):
    fig, ax = plt.subplots(nrows=1, ncols=len(images))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img, cmap="gray")
    if show:
        plt.show()
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight')
        print("{} is saved".format(save_filename))

import cv2
import numpy as np
import scipy.misc
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
