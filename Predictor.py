import os

import cv2

from modules.calibrator import prompt_calibration
from config import CONFIG
from modules.image_processing.converter import convert_img_for_test_or_prediction, convert_img_for_prediction
from modules.image_processing.processor import Processor
from modules.loop import loop
from modules.predictor.predictor import predict
from modules.visualiser.vis import visualise, visualise_prediction
from keras import backend as K
import numpy as np

# Just to specify that the images have to be provided in the model in format (X, Y, channels).
K.set_image_dim_ordering('tf')

size = CONFIG['size']


def fetch_predictor_config():
    try:
        f = open(os.path.join(CONFIG['results_path'], CONFIG['predictor_model_dir'], 'predictor_config.csv'), 'r')
        f.readline()
        values = f.readline().split(',')
        f.close()
        return int(values[0]), str(values[1])
    except:
        return CONFIG['training_img_size'], CONFIG['training_set_image_type']


def predict_action(ip, orig_frame):
    # img_to_predict, img_conversions = convert_img_for_test_or_prediction(ip, orig_frame)
    img_to_predict, img_conversions = convert_img_for_prediction(ip, orig_frame, image_processing_kind, image_size)

    # If the model is trained with shapes (1,50,50), uncomment this line.
    # img_to_predict = np.moveaxis(img_to_predict, -1, 0)

    class_num, normalized_vals, class_name = predict(img_to_predict)

    texts = [
        '~~~~ PREDICTION MODE ~~~~',
        'hsv: ' + str(img_conversions['center_hsv']),
        'predicted: ' + class_name
    ]

    visualise(img_conversions, texts)
    visualise_prediction(normalized_vals, CONFIG['classes'])
    cv2.waitKey(1)


print('Starting predicting mode...')

l_range, u_range = prompt_calibration()
ip = Processor(size, l_range, u_range)

image_size, image_processing_kind = fetch_predictor_config()

loop(predict_action, ip)
cv2.waitKey(0)
