import cv2

from calibrator import prompt_calibration
from config import CONFIG
from image_converter import convert_img_for_test_or_prediction
from image_processing import image_processor
from loop import loop
from training.predictor import predict
from vis import visualise, visualise_prediction

size = CONFIG['size']


def predict_action(ip, orig_frame):
    img_to_predict, img_conversions = convert_img_for_test_or_prediction(ip, orig_frame)

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
ip = image_processor.ImageProcessor(size, l_range, u_range)

loop(predict_action, ip)
cv2.waitKey(0)
