import os
from datetime import datetime

import cv2

from config import CONFIG
from loop import loop
from vis import visualise

size = CONFIG['size']

CLASS = 7

path_to_captured_images = './training/captured_images/'

path_output_dir = path_to_captured_images + str(CLASS) + '/'

last_time = datetime.now()

milliseconds = 200

count = 0

is_capturing = False


def collect_action(frame, frame_with_rect_sq_bboxes, skin, hand, binary_mask, hand_binary_mask, sq_bbox):
    global count, is_capturing

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
            cv2.imwrite(os.path.join(path_output_dir, 'raw_%03d.png') % count, hand)
            count += 1
            print(count)

    params = {
        'count': count
    }

    visualise(params, frame_with_rect_sq_bboxes, skin, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB), hand, hand_binary_mask)


print('Starting data collection mode...')
print('Press "s" to start capturing')

# Take care of folders.
if not os.path.exists(path_output_dir):
    os.makedirs(path_output_dir)

# Loop.
loop(collect_action)

cv2.waitKey(0)
