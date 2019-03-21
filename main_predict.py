import cv2

from config import CONFIG
from image_converter import convert_img_for_test_or_prediction
from loop import loop
from training.predictor import predict
from vis import visualise

size = CONFIG['size']


def predict_action(ip, orig_frame):
    img, img_conversions = convert_img_for_test_or_prediction(ip, orig_frame)

    class_num, normalized_vals, class_name = predict(img)

    def get_center_hsv(img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv = cv2.GaussianBlur(img_hsv, (3, 3), 0)
        center_px_hsv = img_hsv[int(size / 2), int(size / 2)]
        return center_px_hsv

    texts = [
        '~~~~ PREDICTION MODE ~~~~'
        'hsv: ' + str(get_center_hsv(orig_frame)),
        'predicted: ' + class_name
    ]

    visualise(img_conversions, texts)
    cv2.waitKey(1)


print('Starting predicting mode...')

loop(predict_action)
cv2.waitKey(0)
