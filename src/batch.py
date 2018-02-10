# -*- coding: utf-8 -*-

import scipy.misc
import numpy as np
import os
import re
import random
from glob import glob

def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        road_color = np.array([255, 0, 255])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == road_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((np.invert(gt_bg), gt_bg), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


if __name__ == '__main__':
    image_shape = (160, 576)
    get_batches_fn = gen_batch_function('../data/data_road/training', image_shape)
    batch_gen = get_batches_fn(100)
    imgs, gt_imgs = next(batch_gen)
    print(imgs.shape, gt_imgs.shape)
    
    from src.utils import plot_img
    plot_img([imgs[1], gt_imgs[1,:,:,0], gt_imgs[1,:,:,1]])
    
    