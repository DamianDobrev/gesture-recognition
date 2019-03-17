from datetime import datetime
import time

import cv2

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

path_to_captured_images = './captured_images/'

def preprocess_frame(image):
    # TODO Add preprocessing here.
    return image

def capture_video_and_extract_images(class_number, milliseconds=200):
    cap = cv2.VideoCapture(0)

    count = 1
    is_capturing_video = False
    last_time = datetime.now()

    path_output_dir = path_to_captured_images + str(class_number) + '/'

    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('original', frame)

        frame = preprocess_frame(preprocess_frame)
        cv2.imshow('processed', frame)

        if is_capturing_video:
            cur_time = datetime.now()
            time_diff = cur_time - last_time
            time_diff_milliseconds = time_diff.total_seconds() * 1000
            if time_diff_milliseconds >= milliseconds:
                cv2.imwrite(os.path.join(path_output_dir, '%d.png') % count, frame)
                count += 1
                print(count)
        else:
            print('press "s" to start capturing images')

        if cv2.waitKey(1) & 0xFF == ord('s'):
            is_capturing_video = not is_capturing_video

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

capture_video_and_extract_images(2)