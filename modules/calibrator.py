import cv2
import imutils
import numpy as np

from config import CONFIG
from modules.data import fetch_saved_hsv, save_hsv_to_file
from modules.image_processing.converter import get_center_hsv, extract_bounding_boxes_by_skin_threshold
from modules.image_processing.processor import Processor
from modules.image_processing.canvas import Canvas
from modules.visualiser.vis import visualise_orig, append_rectangle_in_center

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


# frame is BGR
def save_ranges(frame):
    global l_h, l_s, l_v, h_h, h_s, h_v, lower_range, upper_range, should_save

    pred_size = 200
    frame = imutils.resize(frame, height=pred_size)
    ip = Processor(size, lower_range, upper_range)
    frame = ip.crop(frame, pred_size)

    h, s, v = get_center_hsv(frame)

    key = cv2.waitKey(5) & 0xFF

    if key == ord('t'):
        should_save = not should_save
    elif key == ord('r'):
        reset_everything()
    elif key == ord('s'):
        save_hsv_to_file(lower_range, upper_range)
    elif key == ord('c'):
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
        '- Press "t" to toggle hsv calibration ON/OFF.',
        '- Press "r" to reset the calibration.',
        '- Press "s" to save current calibration as default.',
        '- Press "c" to confirm current values.',
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


def run_calibrator():
    cap = cv2.VideoCapture(0)
    cv2.destroyAllWindows()

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
    cv2.imshow(CONFIG['imshow_window_name'], canvas.print())

    if cv2.waitKey(0) & 0xFF == ord('c'):
        cv2.destroyAllWindows()

        reset_everything()
        print('started calibration...')
        l_range, u_range = run_calibrator()
        cv2.destroyAllWindows()
        reset_everything()
        print('~~ finished calibration. HSV values:')
        print('  - lower: ', str(l_range))
        print('  - upper: ', str(u_range))

        return l_range, u_range
    else:
        l_range_def, u_range_def = fetch_saved_hsv()
        # l_range_def = [4, 39, 23]
        # u_range_def = [177, 214, 118]

        print('~~ default calibration selected: ')
        print('  - lower: ', str(l_range_def))
        print('  - upper: ', str(u_range_def))

        return l_range_def, u_range_def
