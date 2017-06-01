# !/usr/bin/env python2
# authors: Sartori, Virgili, Zincarini

from __future__ import print_function

import os
import numpy as np
from random import randint

from skimage.io import imread
from sklearn.model_selection import train_test_split
data_path = 'raw/'

image_rows = 240
image_cols = 320
test_percentage = 0.10

def create_train_and_test_data():
    train_data_path = os.path.join(data_path, 'train')
    train_data_path_positive = os.path.join(train_data_path, 'positive')
    train_data_path_partial = os.path.join(train_data_path, 'partial')
    images = os.listdir(train_data_path_positive)
    images.extend(os.listdir(train_data_path_partial))
    total = len(images) / 3

    imgs_16bit = np.ndarray((total, image_rows, image_cols), dtype=np.uint16)
    imgs_8bit = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    ids = []

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
            ids.append(image_name.split('.')[0].replace('_mask', ''))
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

    ids_array = np.array(ids, dtype=object)
    image_position = np.arange(total)

    image_position_train, image_position_test =\
        train_test_split(image_position, test_size=test_percentage, random_state=randint(0, 100))

    ids_train = create_subarray(image_position_train, ids_array)
    ids_test = create_subarray(image_position_test, ids_array)

    imgs_train_16bit = create_subarray(image_position_train, imgs_16bit)
    imgs_test_16bit = create_subarray(image_position_test, imgs_16bit)

    imgs_train_8bit = create_subarray(image_position_train, imgs_8bit)
    imgs_test_8bit = create_subarray(image_position_test, imgs_8bit)

    imgs_mask_train = create_subarray(image_position_train, imgs_mask)
    imgs_mask_test = create_subarray(image_position_test, imgs_mask)

    np.save('imgs_train_16bit.npy', imgs_train_16bit)
    np.save('imgs_train_8bit.npy', imgs_train_8bit)
    np.save('imgs_test_16bit.npy', imgs_test_16bit)
    np.save('imgs_test_8bit.npy', imgs_test_8bit)
    np.save('imgs_train_mask.npy', imgs_mask_train)
    np.save('imgs_test_mask.npy', imgs_mask_test)
    np.save('ids_train.npy', ids_train)
    np.save('ids_test.npy', ids_test)
    print('Saving to .npy files done.')


def create_subarray(subset_index, total_array):
    result_array = []
    for value in subset_index:
        result_array.append(total_array[value])
    return result_array


def load_train_data(bit_image):
    if bit_image == 8:
        imgs_train = np.load('imgs_train_8bit.npy')
    else:
        imgs_train = np.load('imgs_train_16bit.npy')
    imgs_train_mask = np.load('imgs_train_mask.npy')
    imgs_train_id = np.load('ids_train.npy')
    return imgs_train, imgs_train_mask, imgs_train_id


def load_test_data(bit_image):
    if bit_image == 8:
        imgs_test = np.load('imgs_test_8bit.npy')
    else:
        imgs_test = np.load('imgs_test_16bit.npy')
    imgs_test_mask = np.load('imgs_test_mask.npy')
    imgs_test_id = np.load('ids_test.npy')

    return imgs_test, imgs_test_mask, imgs_test_id


if __name__ == '__main__':
    create_train_and_test_data()