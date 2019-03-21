import cv2

import data
from config import CONFIG
from model.model import split_data, create_model, train_model


def run_training():
    # Data fetching
    # ...
    all_img = data.fetch_training_images(CONFIG['path_to_raw'], CONFIG['training_samples'])
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