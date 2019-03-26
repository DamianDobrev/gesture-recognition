import os
from datetime import datetime

import cv2

from calibrator import prompt_calibration
from config import CONFIG
from image_processing.image_converter import convert_img_for_test_or_prediction
from image_processing import image_processor
from loop import loop
from vis import visualise

size = CONFIG['size']

CLASS = CONFIG['class_to_collect_data']

path_output_dir = os.path.join(CONFIG['path_to_raw'], str(CLASS))

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
            cv2.imwrite(os.path.join(path_output_dir, 'raw_%03d.png') % count, img_conversions['hand'])
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
if not os.path.exists(path_output_dir):
    os.makedirs(path_output_dir)

# Calibrate.
l_range, u_range = prompt_calibration()
ip = image_processor.ImageProcessor(size, l_range, u_range)

# Start data collection mode.
loop(collect_action, ip)
cv2.waitKey(0)
