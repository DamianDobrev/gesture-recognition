import csv
import os

import cv2

import numpy as np

import data
from config import CONFIG
from image_converter import convert_img_for_test_or_prediction
from image_processing import image_processor
from model.model import split_data, create_model, train_model

idx = 0


def obtain_hsv_ranges():
    global idx
    f = open(os.path.join('/Users/damian/Desktop/indivpr/training', 'hsv_ranges.csv'), 'r')
    f.readline()
    l_vals = f.readline().split(',')
    u_vals = f.readline().split(',')
    f.close()
    l_range = [int(l_vals[0]), int(l_vals[1]), int(l_vals[2])]
    u_range = [int(u_vals[0]), int(u_vals[1]), int(u_vals[2])]
    print('Processing images for class with index: ' + str(idx))
    idx = idx+1
    return l_range, u_range


def process_tuple(tup):
    images, class_number = tup

    l_range, u_range = obtain_hsv_ranges()
    ip = image_processor.ImageProcessor(CONFIG['size'], l_range, u_range)

    def convert_img_for_test_or_prediction_no_params(img):
        return convert_img_for_test_or_prediction(ip, img)[0]

    return list(map(convert_img_for_test_or_prediction_no_params, images)), class_number


def run_training():
    print('~~~~~ Running training...')

    # Data fetching
    # ...
    all_img = data.fetch_training_images(CONFIG['path_to_raw'], CONFIG['num_training_samples'])
    print('Processing all images as per `convert_img_for_test_or_prediction`...')
    all_img = list(map(process_tuple, all_img))
    print('Processing of images done.')
    x_train, x_test, y_train, y_test = split_data(all_img)

    # Create model
    # ...
    print('num of classes to train: ' + str(len(all_img)))
    model = create_model(len(all_img))
    print('---->>> Training image shape:', x_train[0].shape)

    # Train model
    # ...
    train_model(model, x_train, x_test, y_train, y_test)


run_training()
cv2.waitKey(0)