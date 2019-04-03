import os

import cv2

from modules.calibrator import prompt_calibration
from config import CONFIG
from modules.data import fetch_saved_hsv
from modules.image_processing.converter import convert_img_for_prediction
from modules.image_processing.processor import Processor
from modules.loop import loop
from modules.predictor.predictor import predict
from modules.visualiser.vis import visualise, visualise_prediction
from keras import backend as K

# Just to specify that the images have to be provided in the model in format (X, Y, channels).
K.set_image_dim_ordering('tf')

l_r, u_r = fetch_saved_hsv()
ip_local = Processor(CONFIG['size'], l_r, u_r)


def fetch_predictor_config():
    try:
        f = open(os.path.join(CONFIG['results_path'], CONFIG['predictor_model_dir'], 'predictor_config.csv'), 'r')
        f.readline()
        values = f.readline().split(',')
        f.close()
        return int(values[0]), str(values[1])
    except:
        return CONFIG['training_img_size'], CONFIG['training_set_image_type']


def setup_local_ip():
    global ip_local
    cv2.destroyAllWindows()
    l_range_n, u_range_n = prompt_calibration(True)
    cv2.destroyAllWindows()
    ip_local = Processor(CONFIG['size'], l_range_n, u_range_n)


def predict_action(orig_frame):
    # Initially we need to set this up.
    if ip_local is None:
        setup_local_ip()

    key = cv2.waitKey(5) & 0xFF

    if key == ord('c'):
        setup_local_ip()
    if key == ord('q'):
        exit()

    img_to_predict, img_conversions = convert_img_for_prediction(ip_local, orig_frame, image_processing_kind, image_size)

    # If the model is trained with shapes (1,50,50), uncomment this line.
    # img_to_predict = np.moveaxis(img_to_predict, -1, 0)

    class_num, normalized_vals, class_name = predict(img_to_predict)

    texts = [
        '~~~~ PREDICTION MODE ~~~~',
        '',
        'model directory: ' + str(CONFIG['predictor_model_dir']),
        'center HSV: ' + str(img_conversions['center_hsv']),
        '',
        'predicted label: ' + class_name,
        '',
        'Controls:',
        '- Press "c" to Calibrate',
        '- Press "q" to Quit:'
    ]

    visualise(img_conversions, texts)
    visualise_prediction(normalized_vals, CONFIG['classes'])


print('Starting predicting mode...')

image_size, image_processing_kind = fetch_predictor_config()

loop(predict_action)
cv2.waitKey(0)
