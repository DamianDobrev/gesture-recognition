from datetime import datetime
import time

import cv2
import imutils

from PIL import Image

from keras import backend as K
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
# from sklearn.utils import shuffle
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

import config
import data


import os

from training_scripts.predictor import predict
from training_scripts import image_processor

size = 200
path_to_captured_images = './captured_images/'
path_to_captured_masks = './captured_masks/'

def capture_video_and_extract_images(class_number, milliseconds=200):
    cap = cv2.VideoCapture(0)

    count = 1
    is_saving_images = False
    is_predicting = True
    last_time = datetime.now()

    path_output_dir = path_to_captured_images + str(class_number) + '/'
    path_masks_output_dir = path_to_captured_masks + str(class_number) + '/'

    # ip = image_processor.ImageProcessor(size, [102, 40, 34], [179, 255, 255])  # Works well at home in daylight.
    ip = image_processor.ImageProcessor(size, [104, 25, 34], [179, 255, 180])

    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)
    if not os.path.exists(path_masks_output_dir):
        os.makedirs(path_masks_output_dir)

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, height=size)
        frame = ip.crop(frame)

        processed = ip.extract_skin(frame)

        thr = ip.hsv_to_binary(processed)
        mask_binary = ip.find_largest_connected_component(thr)

        # Find bounding boxes.
        bbox = ip.find_bounding_box_of_binary_img_with_single_component(mask_binary)
        frame_with_bbox = ip.add_bounding_box_to_img(frame, bbox)
        square_bbox = ip.get_square_bbox(bbox, frame)
        frame_with_bboxes = ip.add_bounding_box_to_img(frame_with_bbox, square_bbox, (0, 255, 0))

        # Crop frame to the correct bounding box.
        cropped_image = ip.crop_image_by_square_bbox(frame, square_bbox, size)

        # Also crop the binary mask.
        cropped_binary_mask = ip.crop_image_by_square_bbox(mask_binary, square_bbox, size)
        cropped_binary_mask = cv2.cvtColor(cropped_binary_mask, cv2.COLOR_GRAY2RGB)

        window_name = 'Img with Bbox + processing.'
        # cv2.namedWindow(window_name)
        # cv2.moveWindow(window_name, 40, 40)
        height, width = frame_with_bboxes.shape[:2]
        cv2.rectangle(frame_with_bboxes, (int(width/2-3), int(height/2-3)), (int(width/2 + 3), int(height/2+3)), (255, 255, 0), 1)
        cv2.imshow(window_name, np.hstack([frame_with_bboxes, processed, cropped_image, cropped_binary_mask]))

        if is_predicting:
            predict(cropped_binary_mask)
        elif is_saving_images:
            cur_time = datetime.now()
            time_diff = cur_time - last_time
            time_diff_milliseconds = time_diff.total_seconds() * 1000
            if time_diff_milliseconds >= milliseconds:
                cv2.imwrite(os.path.join(path_output_dir, 'raw_%03d.png') % count, cropped_image)
                cv2.imwrite(os.path.join(path_masks_output_dir, 'raw_%02d.png') % count, cropped_binary_mask)
                count += 1
                print(count)
        else:
            print('S -> start saving images of class [' + str(class_number) + ']')
            print('P -> predict...')

        if cv2.waitKey(1):
            if cv2.waitKey(1) & 0xFF == ord('s'):
                is_saving_images = not is_saving_images
            if cv2.waitKey(1) & 0xFF == ord('b'):
                is_predicting = not is_predicting
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

capture_video_and_extract_images(999)
cv2.waitKey(0)