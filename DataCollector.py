import getopt
import os
import sys
from datetime import datetime

import cv2

from modules.calibrator import prompt_calibration
from config import CONFIG
from modules.data import fetch_saved_hsv
from modules.image_processing.converter import convert_image, augment_image
from modules.image_processing.processor import Processor
from modules.loop import loop
from modules.visualiser.vis import visualise

size = CONFIG['size']

CLASS = CONFIG['data_collect_class']

tr_path = CONFIG['training_data_path']
tr_name = CONFIG['data_collect_set_name']

output_dir = os.path.join(tr_path, tr_name)

last_time = datetime.now()

milliseconds = 200

count = 0

is_capturing = False

l_r, u_r = fetch_saved_hsv()
ip = Processor(CONFIG['size'], l_r, u_r)


def setup_image_processor():
    global ip
    # Calibrate.
    l_range, u_range = prompt_calibration(True)
    ip = Processor(size, l_range, u_range)
    cv2.destroyAllWindows()


def collect_action(frame):
    global count, is_capturing, ip

    if ip is None:
        setup_image_processor()

    img_conversions = convert_image(ip, frame)

    key = cv2.waitKey(5) & 0xFF

    if key == ord('c'):
        setup_image_processor()
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
        if time_diff_milliseconds >= milliseconds:
            # Here we iterate over all preprocessed images from the list and save
            # them into separate folders.
            imgs = ['orig', 'orig_monochrome', 'skin_monochrome', 'hand', 'hand_binary_mask']
            for key in imgs:
                file_output_dir = os.path.join(output_dir, key, str(CLASS))
                if not os.path.exists(file_output_dir):
                    os.makedirs(file_output_dir)
                cv2.imwrite(os.path.join(file_output_dir, '%04d.png') % count, img_conversions[key])

            # Augmentation.
            aug_folder_name = 'skin_monochrome_augmented'
            file_output_dir = os.path.join(output_dir, aug_folder_name, str(CLASS))
            if not os.path.exists(file_output_dir):
                os.makedirs(file_output_dir)

            augmented_images = augment_image(img_conversions['skin_monochrome'])

            for im in augmented_images:
                cv2.imwrite(os.path.join(file_output_dir, '%04d.png') % count, im)

            count += 1
            print(count)

    classes = CONFIG['classes']
    class_with_idx_in_config = classes[CLASS-1] if len(classes) < CLASS-1 else '!!! None !!!'
    texts = [
        '~~~~ DATA COLLECTION MODE ~~~~',
        '',
        'Class idx: ' + str(CLASS),
        'Class with idx in CONFIG: ' + class_with_idx_in_config,
        'Saving to dir: ' + os.path.join(CONFIG['training_data_path'], CONFIG['data_collect_set_name']),
        '',
        'Num of collected imgs: ' + str(count),
        '',
        '',
        'Controls:',
        '- Press "s" to Stop capturing' if is_capturing else 'press "s" to Start capturing',
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
        CONFIG['data_collect_class'] = arg

print('Starting data collection mode...')
print('Press "s" to start capturing')

# Take care of folders.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Start data collection mode.
loop(collect_action)
# cv2.waitKey(0)
