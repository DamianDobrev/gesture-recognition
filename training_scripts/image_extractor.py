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

from training_scripts import image_processor

size = 200
path_to_captured_images = './captured_images/'

def crop(img):
    frame = img[0:200, 100:300]
    return frame

def capture_video_and_extract_images(class_number, milliseconds=200):
    cap = cv2.VideoCapture(0)

    count = 1
    is_saving_images = False
    last_time = datetime.now()

    path_output_dir = path_to_captured_images + str(class_number) + '/'

    ip = image_processor.ImageProcessor([120, 24, 34], [179, 255, 255])

    if not os.path.exists(path_output_dir):
        os.makedirs(path_output_dir)

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=size)
        frame = crop(frame)

        processed = ip.preprocess_frame(frame)
        # edges = ip.edges(frame)
        cv2.imshow('processed', np.vstack([frame, processed]))

        thr = ip.in_binary(processed)
        cv2.imshow('shit', thr)

        if is_saving_images:
            cur_time = datetime.now()
            time_diff = cur_time - last_time
            time_diff_milliseconds = time_diff.total_seconds() * 1000
            if time_diff_milliseconds >= milliseconds:
                cv2.imwrite(os.path.join(path_output_dir, 'raw_%02d.png') % count, frame)
                count += 1
                print(count)
        else:
            print('press "s" to start capturing images')

        if cv2.waitKey(1) & 0xFF == ord('s'):
            is_saving_images = not is_saving_images

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

capture_video_and_extract_images(2)
cv2.waitKey(0)