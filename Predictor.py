import cv2

from modules.calibrator import prompt_calibration
from config import CONFIG
from modules.image_processing.converter import convert_img_for_test_or_prediction
from modules.image_processing.processor import Processor
from modules.loop import loop
from modules.predictor.predictor import predict
from modules.visualiser.vis import visualise, visualise_prediction
from keras import backend as K
import numpy as np

# Just to specify that the images have to be provided in the model in format (X, Y, channels).
K.set_image_dim_ordering('tf')

size = CONFIG['size']


def predict_action(ip, orig_frame):
    img_to_predict, img_conversions = convert_img_for_test_or_prediction(ip, orig_frame)

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

loop(predict_action, ip)
cv2.waitKey(0)
