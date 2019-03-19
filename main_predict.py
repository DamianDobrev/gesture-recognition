import cv2

from config import CONFIG
from loop import loop
from training.predictor import predict
from vis import visualise

size = CONFIG['size']


def predict_action(frame, frame_with_rect_sq_bboxes, skin, hand, binary_mask, hand_binary_mask, sq_bbox):
    class_num, normalized_vals, class_name = predict(hand_binary_mask)

    def get_center_hsv(img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv = cv2.GaussianBlur(img_hsv, (3, 3), 0)
        center_px_hsv = img_hsv[int(size / 2), int(size / 2)]
        return center_px_hsv

    params = {
        'center_hsv': get_center_hsv(frame),
        'result': class_name
    }

    visualise(params, frame_with_rect_sq_bboxes, skin, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB), hand, hand_binary_mask)
    cv2.waitKey(1)


print('Starting predicting mode...')

loop(predict_action)
cv2.waitKey(0)
