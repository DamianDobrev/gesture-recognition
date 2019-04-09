import cv2
import imutils
import numpy as np

from config import CONFIG
from modules.data import fetch_saved_hsv, save_hsv_to_file
from modules.image_processing.converter import get_center_hsv, extract_bounding_boxes_by_skin_threshold
import modules.image_processing.processor as imp
from modules.loop import loop_camera_frames
from modules.visualiser.visualiser import visualise_calibration, draw_rectangle_in_center

size = CONFIG['size']


def reset_global_values(l_r=None, u_r=None):
    """
    Resets the global values for lower and upper skin color ranges and sets
    the "should save" parameter to False.
    :param l_r: Initial lower skin color threshold of type [h,s,v].
    :param u_r: Initial upper skin color threshold of type [h,s,v].
    :return:
    """
    global l_h, l_s, l_v, h_h, h_s, h_v, lower_range, upper_range, should_save
    # Cannot use default args here because they will be mutable, which is dangerous.
    lower_range = l_r if l_r is not None else np.array([100, 100, 100])
    upper_range = u_r if u_r is not None else np.array([101, 101, 101])

    l_h = lower_range[0]
    l_s = lower_range[1]
    l_v = lower_range[2]
    h_h = upper_range[0]
    h_s = upper_range[1]
    h_v = upper_range[2]

    should_save = False


def calibrate(frame):
    """
    Calibrate the lower and upper skin threshold values.
    :param frame: The current frame taken from the web camera. It is
        a BGR image with shape (X,Y,3).
    :return: Does not return anything.
    """
    global l_h, l_s, l_v, h_h, h_s, h_v, lower_range, upper_range, should_save

    pred_size = CONFIG['size']
    frame = imutils.resize(frame, height=pred_size)
    frame = imp.crop_from_center(frame, pred_size)

    h, s, v = get_center_hsv(frame)

    key = cv2.waitKey(5) & 0xFF

    if key == ord('t'):
        should_save = not should_save
    elif key == ord('r'):
        reset_global_values()
    elif key == ord('s'):
        save_hsv_to_file(lower_range, upper_range)
    elif key == ord('c'):
        return True
    elif key == ord('q'):
        return exit()

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
            lower_range = np.array([l_h, l_s, l_v])
            upper_range = np.array([h_h, h_s, h_v])

    texts = [
        '~~~~ CALIBRATION ~~~~',
        'is_saving:' + str(should_save),
        '',
        'Put your hand in the middle and move it so that the ',
        'rectangle captures all the shades of your skin color.',
        '- Press "t" to Toggle hsv calibration ON/OFF.',
        '- Press "r" to Reset the calibration.',
        '- Press "s" to Save current calibration as default.',
        '',
        '- Press "c" to Confirm current values.',
        '- Press "q" to Quit.',
        '',
        'hsv: ' + str([h, s, v]),
        'lower: ' + str(lower_range),
        'upper: ' + str(upper_range),
    ]

    skin, binary_mask, bbox, sq_bbox = extract_bounding_boxes_by_skin_threshold(frame, lower_range, upper_range)

    binary_mask = imp.isolate_largest_connected_component(binary_mask)
    binary_mask = draw_rectangle_in_center(binary_mask)
    frame = draw_rectangle_in_center(frame)

    binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    visualise_calibration(np.hstack([frame, binary_mask, skin]), texts, 'Calibrator')


def run_calibrator():
    """
    Runs video capture and visualizes the image processing based on the current calibration.
    Calibration can be changed via the UI controls.
    :return:
    """
    cap = cv2.VideoCapture(0)
    cap.set(15, 0.00001)
    cv2.destroyAllWindows()

    loop_camera_frames(calibrate)

    cap.release()
    cv2.destroyAllWindows()
    return lower_range, upper_range


def prompt_calibration():
    """
    Initializes and runs the calibrator.
    It initially resets the values based on the current values in the CONFIG['hsv_ranges_path'] file.
    :return: The new HSV thresholds:
        - Lower skin color threshold of type [h,s,v] (as calibrated).
        - Upper skin color threshold of type [h,s,v] (as calibrated).
    """
    global lower_range, upper_range

    # Initialize.
    cv2.destroyAllWindows()
    saved_l_r, saved_u_r = fetch_saved_hsv()
    reset_global_values(saved_l_r, saved_u_r)
    print('started calibration...')

    # Calibrate.
    l_range, u_range = run_calibrator()
    cv2.destroyAllWindows()
    reset_global_values()

    print('~~ finished calibration. HSV values:')
    print('  - lower: ', str(l_range))
    print('  - upper: ', str(u_range))

    return l_range, u_range
