import os
import time
from datetime import datetime

import cv2

from modules.calibrator import prompt_calibration
from config import CONFIG
from modules.image_processing.converter import convert_img_for_test_or_prediction
from modules.image_processing.processor import Processor
from modules.loop import loop
from modules.visualiser.vis import visualise

size = CONFIG['size']

CLASS = CONFIG['class_to_collect_data']

tr_path = CONFIG['training_sets_path']
tr_name = CONFIG['training_set_name']

output_dir = os.path.join(tr_path, tr_name)

# st = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d__%H-%M-%S')

last_time = datetime.now()

milliseconds = 200

count = 0

is_capturing = False


def collect_action(ip, frame):
    global count, is_capturing

    img, img_conversions = convert_img_for_test_or_prediction(ip, frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        is_capturing = not is_capturing
        if not is_capturing:
            print('Press "s" to start capturing')
        else:
            print('Started capturing...')

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
            count += 1
            print(count)

    texts = [
        '~~~~ DATA COLLECTION MODE ~~~~',
        'class: ' + str(CLASS),
        'count: ' + str(count),
        'press "s" to stop capturing' if is_capturing else 'press "s" to start capturing'
    ]

    visualise(img_conversions, texts)


print('Starting data collection mode...')
print('Press "s" to start capturing')

# Take care of folders.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Calibrate.
l_range, u_range = prompt_calibration()
ip = Processor(size, l_range, u_range)
cv2.destroyAllWindows()

# Start data collection mode.
loop(collect_action, ip)
# cv2.waitKey(0)
