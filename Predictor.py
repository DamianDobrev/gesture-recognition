# Run this file with following commands:
# "-m predictor_model_dir" Defaults to the one in config.py.
# "-r" if specified, uses the RDS mapping to fire keyboard events. Defaults to False.

import getopt
import os
import sys

import cv2

from modules.calibrator import prompt_calibration
from config import CONFIG
from modules.data import fetch_saved_hsv
from modules.image_processing.converter import convert_img_for_prediction
from modules.loop import loop_camera_frames
from modules.predictor.predictor import predict
from modules.simulator.Sim import Simulator
from modules.visualiser.vis import visualise, visualise_prediction_result
from keras import backend as K

# Just to specify that the images have to be provided in the model in format (X, Y, channels).
K.set_image_dim_ordering('tf')

l_hsv_thresh, u_hsv_thresh = fetch_saved_hsv()

simulator = Simulator()


def fetch_predictor_config():
    """
    Fetches the metadata of the trained model in order to be able to prepare
    images the same way as the trained model expects them.
    :return:
    """
    try:
        f = open(os.path.join(CONFIG['results_path'], CONFIG['predictor_model_dir'], 'config.csv'), 'r')
        f.readline()
        values = f.readline().split(',')
        f.close()
        return int(values[0]), str(values[2]).strip()
    except:
        # If this happens, check the path.
        print('ERROR in fetching predictor config. Default values used. These may fail.')
        return CONFIG['training_img_size'], CONFIG['training_set_image_type']


def setup_hsv_boundaries():
    """
    Sets up the global lower and upper boundaries of the skin color by running
    the calibrator.
    :return:
    """
    global l_hsv_thresh, u_hsv_thresh
    cv2.destroyAllWindows()
    l_hsv_thresh, u_hsv_thresh = prompt_calibration()
    cv2.destroyAllWindows()


def predict_gesture_and_visualise_result(raw_image):
    """
    Predicts the gesture on the raw_image and visualizes the result.
    If CONFIG['simulator_on'] is set to True, it also sends commands to RDS.
    :param raw_image: A BGR image of shape (X,Y,3).
    :return: Does not return anything.
    """
    key = cv2.waitKey(5) & 0xFF

    if key == ord('c'):
        setup_hsv_boundaries()
    if key == ord('q'):
        exit()

    img_to_predict, img_conversions = convert_img_for_prediction(raw_image, l_hsv_thresh, u_hsv_thresh,
                                                                 image_processing_kind, image_size)

    # If the model is trained with shapes (1,50,50), uncomment this line.
    # img_to_predict = np.moveaxis(img_to_predict, -1, 0)

    class_num, normalized_vals, class_name = predict(img_to_predict)

    texts = [
        '~~~~ PREDICTION MODE ~~~~',
        '',
        'model directory: ' + str(CONFIG['predictor_model_dir']),
        'predicted label: ' + class_name,
        '',
        'Controls:',
        '- Press "c" to Calibrate',
        '- Press "q" to Quit:'
    ]

    coy = img_conversions['center_offset_y']
    cox = img_conversions['center_offset_x']
    # This number provides an offset on each side, that should account for bounding box being of some size.
    visualise_prediction_result(normalized_vals, CONFIG['classes'], cox, coy, CONFIG['size'] - 100)
    visualise(img_conversions, texts)

    simulator.perform_action(class_name, cox, coy)


# Setup config to use args.
opts, args = getopt.getopt(sys.argv[1:],
                           "m:s")

for opt, arg in opts:
    opt = opt.strip()
    if opt == '-m':
        CONFIG['predictor_model_dir'] = arg
    if opt == '-s':
        CONFIG['simulator_on'] = True

print('Starting predicting mode...')

# This fetches the configuration from the training folder. This way
# we know which preprocessing technique to feed the model with.
image_size, image_processing_kind = fetch_predictor_config()

loop_camera_frames(predict_gesture_and_visualise_result)
cv2.waitKey(0)
