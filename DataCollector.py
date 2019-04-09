# Run this file with following commands:
# -n data_collect_set_name
# -c data_collect_class (number in range 1..(len(CONFIG['classes'])+1).
# If this is not provided, default values are taken from config.py.

import getopt
import os
import sys
from datetime import datetime

import cv2

from modules.calibrator import prompt_calibration
from config import CONFIG
from modules.data import fetch_saved_hsv
from modules.image_processing.converter import convert_image, augment_image
from modules.loop import loop_camera_frames
from modules.visualiser.visualiser import visualise

size = CONFIG['size']
CLASS = CONFIG['data_collect_class']
training_data_path = CONFIG['training_data_path']
data_collect_set_name = CONFIG['data_collect_set_name']
output_dir = os.path.join(training_data_path, data_collect_set_name)

last_time = datetime.now()
save_file_interval_ms = 200
current_count = 0
is_capturing = False

l_hsv_bound, u_hsv_bound = fetch_saved_hsv()


def calibrate():
    """
    Sets up global hsv boundaries for skin color.
    :return:
    """
    global l_hsv_bound, u_hsv_bound
    l_hsv_bound, u_hsv_bound = prompt_calibration()
    cv2.destroyAllWindows()


def collect_action(frame):
    """
    Visualizes current frame and preprocessing.
    If it is capturing, saves the processed images to the appropriate path.
    There is a UI:
    - Pressing C starts/stops calibration mode.
    - Pressing S starts/stops image saving.
    :param frame: A frame taken by web cam during video.
    :return: Does not return anything.
    """
    global current_count, is_capturing

    img_conversions = convert_image(frame, l_hsv_bound, u_hsv_bound)

    key = cv2.waitKey(5) & 0xFF

    if key == ord('c'):
        calibrate()
    if key == ord('s'):
        is_capturing = not is_capturing
        if not is_capturing:
            print('Press "s" to start capturing')
        else:
            print('Started capturing...')
    if key == ord('q'):
        exit()

    if is_capturing:
        cur_time = datetime.now()
        time_diff = cur_time - last_time
        time_diff_milliseconds = time_diff.total_seconds() * 1000
        if time_diff_milliseconds >= save_file_interval_ms:
            # Here we iterate over all preprocessed images from the list and save
            # them into separate folders.
            imgs = ['orig', 'orig_monochrome', 'skin_monochrome', 'hand', 'hand_binary_mask']
            for key in imgs:
                file_output_dir = os.path.join(output_dir, key, str(CLASS))
                if not os.path.exists(file_output_dir):
                    os.makedirs(file_output_dir)
                cv2.imwrite(os.path.join(file_output_dir, '%04d.png') % current_count, img_conversions[key])

            # Augmentation.
            aug_folder_name = 'skin_monochrome_augmented'
            file_output_dir = os.path.join(output_dir, aug_folder_name, str(CLASS))
            if not os.path.exists(file_output_dir):
                os.makedirs(file_output_dir)

            augmented_images = augment_image(img_conversions['skin_monochrome'])

            for im in augmented_images:
                cv2.imwrite(os.path.join(file_output_dir, '%04d.png') % current_count, im)

            current_count += 1
            print(current_count)

    classes = CONFIG['classes']
    class_with_idx_in_config = classes[CLASS-1] if len(classes) < CLASS-1 else '!!! None !!!'
    texts = [
        '~~~~ DATA COLLECTION MODE ~~~~ ' + '!!!CAPTURING!!!' if is_capturing else '',
        'Class with idx ' + str(CLASS) + ' named: ' + class_with_idx_in_config,
        'Saving to dir: ' + os.path.join(CONFIG['training_data_path'], CONFIG['data_collect_set_name']),
        'Num of collected imgs: ' + str(current_count),
        '',
        'Controls:',
        '- Press "s" to Stop capturing' if is_capturing else '- Press "s" to Start capturing',
        '- Press "c" to Calibrate',
        '- Press "q" to Quit:'
    ]

    visualise(img_conversions, texts)


# Setup config to use args.
opts, args = getopt.getopt(sys.argv[1:],
                           "n:c:",
                           ["data_collect_set_name=", "data_collect_class"])

for opt, arg in opts:
    if opt in ('-n', '--data_collect_set_name'):
        CONFIG['data_collect_set_name'] = arg
    if opt in ('-c', '--data_collect_class'):
        CONFIG['data_collect_class'] = int(arg)

print('Starting data collection mode...')
print('Press "s" to start capturing')

# Take care of folders' existence.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Start data collection mode.
loop_camera_frames(collect_action)
