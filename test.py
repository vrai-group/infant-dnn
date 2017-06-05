from __future__ import print_function

import os
import datetime
from skimage.io import imsave
import numpy as np
from keras import backend as K
from data import load_train_data, load_test_data
from train import preprocess, evaluate_result, get_unet

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

trained_model_path = 'trained model'

def test_model(bit_image):
    imgs_train, imgs_mask_train, imgs_train_ids = load_trained_model(bit_image)
    imgs_train = preprocess(imgs_train)
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test, imgs_mask_test, imgs_id_test = load_test_data(bit_image)
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model = get_unet()
    model.load_weights(os.path.join(trained_model_path,'weights.h5'))

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    imgs_mask_pred = model.predict(imgs_test, verbose=1)

    print('-' * 30)
    print('Calculating accuracy...')
    print('-' * 30)
    evaluate_result(imgs_mask_pred, imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds_' + str(datetime.datetime.now()).replace(":","-")
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    np.save(os.path.join(pred_dir,'imgs_mask_pred.npy'), imgs_mask_pred)
    for image, image_id in zip(imgs_mask_pred, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

def load_trained_model(bit_image):
    if bit_image == 8:
        imgs_train = np.load(os.path.join(trained_model_path,'imgs_train_8bit.npy'))
    else:
        imgs_train = np.load(os.path.join(trained_model_path,'imgs_train_16bit.npy'))
    imgs_train_mask = np.load(os.path.join(trained_model_path,'imgs_train_mask.npy'))
    imgs_train_id = np.load(os.path.join(trained_model_path,'ids_train.npy'))
    return imgs_train, imgs_train_mask, imgs_train_id

if __name__ == '__main__':
    test_model(8)
    #test_model(16)