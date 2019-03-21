import os
from datetime import datetime

import cv2
import imutils
import numpy as np

from config import CONFIG
from data import fetch_training_images
from image_converter import convert_image
from image_processing import image_processor

size = CONFIG['size']
cap = cv2.VideoCapture(0)
last_time = datetime.now()

path_to_captured_images = './training/captured_images/'
path_to_captured_masks = './training/captured_masks/'

# ip = image_processor.ImageProcessor(size, [102, 40, 34], [179, 255, 255])  # Works well at home in daylight.
# ip = image_processor.ImageProcessor(size, [95, 90, 150], [179, 255, 255])
ip = image_processor.ImageProcessor(size, [104, 25, 34], [179, 255, 180])  # Uni.


def loop(fn, img_processor=ip):
    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, height=size)
        frame = ip.crop(frame)
        should_break = fn(img_processor, frame)
        if should_break:
            break

    cap.release()
    cv2.destroyAllWindows()


# def add_folder():
#     if not os.path.exists(path_to_captured_masks):
#         os.makedirs(path_to_captured_masks)
#
#     imgs = fetch_training_images(path_to_captured_images, 700)
#     for i in range(0,7):
#         img_list = imgs[i][0]
#         class_num = imgs[i][1]
#         print(np.array(img_list).shape)
#         print(class_num)
#
#         for j, img in enumerate(img_list):
#             # img = np.moveaxis(img, -1, 0)
#             # img = np.moveaxis(img, -1, 0)
#             # print('dasdsa', img.shape)
#             # cv2.imshow('shittty', img)
#             # cv2.waitKey(0)
#             # print('kur', path_to_captured_masks + str(class_num) + '/')
#             def save_masked_img(img, frame_with_rect_sq_bboxes, skin, hand, binary_mask, hand_binary_mask, sq_bbox):
#                 print('kur', path_to_captured_masks + str(class_num + 1) + '/')
#                 path = path_to_captured_masks + str(class_num + 1) + '/'
#                 if not os.path.exists(path):
#                     os.makedirs(path)
#
#                 cv2.imwrite(os.path.join(path, 'binary_%03d.png') % (j+1), hand_binary_mask)
#             run(img, save_masked_img)

# add_folder()