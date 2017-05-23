# !/usr/bin/env python2
# authors: Sartori, Virgili, Zincarini

from __future__ import print_function

import os
import numpy as np

from skimage.io import imread

data_path = 'raw/'

image_rows = 240
image_cols = 320


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    train_data_path_positive = os.path.join(train_data_path, 'positive')
    train_data_path_partial = os.path.join(train_data_path, 'partial')
    images = os.listdir(train_data_path_positive)
    images.extend(os.listdir(train_data_path_partial))
    total = len(images) / 3

    imgs_16bit = np.ndarray((total, image_rows, image_cols), dtype=np.uint16)
    imgs_8bit = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    counter_16bit = 0
    counter_8bit = 0
    counter_mask = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        path = train_data_path_positive
        if os.path.isfile(os.path.join(train_data_path_partial, image_name)):
            path = train_data_path_partial

        if 'mask' in image_name:
            img_mask = imread(os.path.join(path, image_name), as_grey=True)
            img_mask = np.array([img_mask])
            imgs_mask[counter_mask] = img_mask
            counter_mask += 1
        else:
            img = imread(os.path.join(path, image_name), as_grey=True)
            img = np.array([img])
            if image_name.split('.')[0].endswith('_8'):
                imgs_8bit[counter_8bit] = img
                counter_8bit += 1
            else:
                imgs_16bit[counter_16bit] = img
                counter_16bit += 1

        if counter_mask % 100 == 0 and counter_8bit % 100 == 0 and counter_16bit % 100 == 0:
            print('Done: {0}/{1} images'.format(counter_mask, total))
        elif counter_mask == total-1 and counter_8bit == total-1 and counter_16bit == total-1:
            print('Done: {0}/{1} images'.format(total, total))
    print('Loading done.')

    np.save('imgs_train_16bit.npy', imgs_16bit)
    np.save('imgs_train_8bit.npy', imgs_8bit)
    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')

if __name__ == '__main__':
    create_train_data()