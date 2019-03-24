import time

import cv2
import imutils
import numpy as np

from config import CONFIG
from image_converter import convert_img_for_test_or_prediction, get_center_hsv, extract_bounding_boxes_by_skin_threshold
from image_processing import image_processor
from image_processing.canvas import Canvas
from vis import visualise_orig, append_rectangle_in_center

size = CONFIG['size']

init_lower_range = [100, 100, 100]
init_upper_range = [101, 101, 101]


def reset_everything():
    global l_h, l_s, l_v, h_h, h_s, h_v, lower_range, upper_range, should_save
    lower_range = init_lower_range
    upper_range = init_upper_range

    l_h = None
    l_s = None
    l_v = None
    h_h = None
    h_s = None
    h_v = None

    should_save = False


def normalize_hist(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to BGR format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


# frame is BGR
def save_ranges(frame):
    global l_h, l_s, l_v, h_h, h_s, h_v, lower_range, upper_range, should_save

    pred_size = 200
    frame = imutils.resize(frame, height=pred_size)
    ip = image_processor.ImageProcessor(size, lower_range, upper_range)
    frame = ip.crop(frame, pred_size)

    h, s, v = get_center_hsv(frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        should_save = not should_save

    if cv2.waitKey(1) & 0xFF == ord('r'):
        return True

    if cv2.waitKey(1) & 0xFF == ord('c'):
        return True

    if should_save:
        if l_h is None or h < l_h:
            l_h = h
        elif h_h is None or h > h_h:
            h_h = h

        if l_s is None or s < l_s:
            l_s = s
        elif h_s is None or s > h_s:
            h_s = s

        if l_v is None or v < l_v:
            l_v = v
        elif h_v is None or v > h_v:
            h_v = v

        if l_h is not None and l_s is not None and l_v is not None and h_h is not None and h_s is not None and h_v:
            lower_range = [l_h, l_s, l_v]
            upper_range = [h_h, h_s, h_v]

    texts = [
        '~~~~ CALIBRATION ~~~~',
        'Put your hand in the middle and move it so that the ',
        'rectangle captures all the shades of your skin color.',
        '- Press "s" to start/stop hsv saving.',
        '- Press "c" to confirm current values.',
        '- Press "r" to restart current values.',
        '',
        'hsv: ' + str([h, s, v]),
        'lower: ' + str(lower_range),
        'upper: ' + str(upper_range),
        'is_saving:' + str(should_save),
    ]

    skin, binary_mask, bbox, sq_bbox = extract_bounding_boxes_by_skin_threshold(ip, frame)

    binary_mask = ip.find_largest_connected_component(binary_mask)
    binary_mask = append_rectangle_in_center(binary_mask)
    frame = append_rectangle_in_center(frame)

    binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    visualise_orig(np.hstack([frame, binary_mask, skin]), texts)

    cv2.waitKey(1)


def run_calibrator():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        should_break = save_ranges(frame)
        if should_break:
            break

    cap.release()
    cv2.destroyAllWindows()
    return lower_range, upper_range


def prompt_calibration():
    canvas = Canvas((100, 800, 3))
    text = 'To calibrate HSV values press "c", to use default calibration press any other key.'
    canvas.draw_text(1, text)
    cv2.imshow('calibration', canvas.print())

    if cv2.waitKey(0) & 0xFF == ord('c'):
        cv2.destroyAllWindows()

        reset_everything()
        print('~~ started calibration')
        l_range, u_range = run_calibrator()
        reset_everything()
        print('~~ finished calibration. HSV values:')
        print('  - lower: ', str(l_range))
        print('  - upper: ', str(u_range))

        return l_range, u_range
    else:
        # Best config for home lamps:
        # - lower: [3, 37, 64]
        # - upper: [177, 168, 154]
        l_range_def = [3, 37, 64]
        u_range_def = [177, 168, 154]

        print('default calibration selected: ')
        print('  - lower: ', str(l_range_def))
        print('  - upper: ', str(u_range_def))

        return l_range_def, u_range_def
