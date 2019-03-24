import cv2

import numpy as np

import data
from config import CONFIG
from image_processing.image_processor import to_50x50_monochrome
from model.model import split_data, create_model, train_model


def process_tuple(tup):
    images, class_number = tup
    return list(map(to_50x50_monochrome, images)), class_number


def run_training():
    # Data fetching
    # ...
    all_img = data.fetch_training_images(CONFIG['path_to_raw'], CONFIG['num_training_samples'])
    all_img = list(map(process_tuple, all_img))
    x_train, x_test, y_train, y_test = split_data(all_img)

    # Create model
    # ...
    model = create_model(len(all_img))
    print('---->>> Training image shape:', x_train[0].shape)

    # Train model
    # ...
    train_model(model, x_train, x_test, y_train, y_test)


run_training()
cv2.waitKey(0)