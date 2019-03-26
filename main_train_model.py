import cv2

import data
from config import CONFIG
from image_converter import convert_img_for_test_or_prediction
from image_processing import image_processor
from model.model import split_data, create_model, train_model

idx = 0


def process_tuple(tup):
    images, class_number = tup

    l_range, u_range = data.fetch_saved_hsv()
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
    cv2.destroyAllWindows()
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