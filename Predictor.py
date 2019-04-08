import getopt
import os
import sys

import cv2

from modules.calibrator import prompt_calibration
from config import CONFIG
from modules.data import fetch_saved_hsv
from modules.image_processing.converter import convert_img_for_prediction
from modules.loop import loop
from modules.predictor.predictor import predict
from modules.simulator.simulator import RDS
from modules.visualiser.vis import visualise, visualise_prediction
from keras import backend as K

# Just to specify that the images have to be provided in the model in format (X, Y, channels).
K.set_image_dim_ordering('tf')

l_hsv_thresh, u_hsv_thresh = fetch_saved_hsv()

rds = RDS()


def fetch_predictor_config():
    try:
        f = open(os.path.join(CONFIG['results_path'], CONFIG['predictor_model_dir'], 'config.csv'), 'r')
        f.readline()
        values = f.readline().split(',')
        f.close()
        return int(values[0]), str(values[2]).strip()
    except:
        print('ERROR in fetching predictor config. Default values used.')
        return CONFIG['training_img_size'], CONFIG['training_set_image_type']


def setup_hsv_boundaries():
    global l_hsv_thresh, u_hsv_thresh
    cv2.destroyAllWindows()
    l_hsv_thresh, u_hsv_thresh = prompt_calibration(True)
    cv2.destroyAllWindows()


def predict_action(orig_frame):
    key = cv2.waitKey(5) & 0xFF

    if key == ord('c'):
        setup_hsv_boundaries()
    if key == ord('q'):
        exit()

    print('l_hsv_thresh',l_hsv_thresh)
    print('u_hsv_thresh',u_hsv_thresh)
    img_to_predict, img_conversions = convert_img_for_prediction(orig_frame, l_hsv_thresh, u_hsv_thresh,
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
    visualise_prediction(normalized_vals, CONFIG['classes'], cox, coy, CONFIG['size'] - 100)
    visualise(img_conversions, texts)

    if CONFIG['RDS']:
        rds.do(class_name, cox, coy)


# Setup config to use args.
opts, args = getopt.getopt(sys.argv[1:],
                           "m:r",
                           ["predictor_model_dir="])

for opt, arg in opts:
    if opt in ('-m', '--predictor_model_dir'):
        CONFIG['predictor_model_dir'] = arg
    if opt in ('-r', '--RDS'):
        CONFIG['RDS'] = True

print('Starting predicting mode...')

image_size, image_processing_kind = fetch_predictor_config()

loop(predict_action)
cv2.waitKey(0)
