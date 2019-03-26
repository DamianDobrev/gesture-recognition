import os

import cv2

from modules import data
from config import CONFIG
# from modules.image_processing.converter import convert_img_for_test_or_prediction
from modules.image_processing.processor import resize_to_training_img_size, convert_to_one_channel_monochrome
from modules.model.model import split_data, create_model, train_model


def correct_images_shapes_in_tuple(tup):
    images, class_number = tup

    def convert_img_for_test_or_prediction_no_params(img):
        return convert_to_one_channel_monochrome(resize_to_training_img_size(img))

    return list(map(convert_img_for_test_or_prediction_no_params, images)), class_number


def run_training():
    print('~~~~~ Running training...')

    # Data fetching
    # ...
    set_path = CONFIG['training_sets_path']
    set_name = CONFIG['training_set_name']
    set_type = CONFIG['training_set_image_type']
    path = os.path.join(set_path, set_name, set_type)
    all_img = data.fetch_training_images(path, CONFIG['num_training_samples'])
    print('Processing all images to match the shape expected by the model...')
    all_img = list(map(correct_images_shapes_in_tuple, all_img))
    cv2.destroyAllWindows()
    print('Processing of images done.')
    x_train, x_test, y_train, y_test = split_data(all_img)

    # Create model
    # ...
    print('num of classes to train: ' + str(len(all_img)))
    model = create_model(len(all_img))
    print('---->>> Training image shape:', x_train[0].shape)
    cv2.imshow('sampleimg', x_train[0])
    cv2.waitKey(1)

    # Train model
    # ...
    train_model(model, x_train, x_test, y_train, y_test)
    cv2.destroyAllWindows()


run_training()
cv2.waitKey(0)