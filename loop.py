import os
from datetime import datetime

import cv2
import imutils
import numpy as np

from config import CONFIG
from data import fetch_training_images_binary
from image_processing import image_processor

size = CONFIG['size']
cap = cv2.VideoCapture(0)
last_time = datetime.now()

path_to_captured_images = './training/captured_images/'
path_to_captured_masks = './training/captured_masks/'

# ip = image_processor.ImageProcessor(size, [102, 40, 34], [179, 255, 255])  # Works well at home in daylight.
ip = image_processor.ImageProcessor(size, [95, 90, 150], [179, 255, 255])  # Works well at home in daylight.
# ip = image_processor.ImageProcessor(size, [104, 25, 34], [179, 255, 180])


def loop(fn):
    # path_output_dir = path_to_captured_images + str(class_number) + '/'
    # path_masks_output_dir = path_to_captured_masks + str(class_number) + '/'

    # if not os.path.exists(path_output_dir):
    #     os.makedirs(path_output_dir)
    # if not os.path.exists(path_masks_output_dir):
    #     os.makedirs(path_masks_output_dir)

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, height=size)
        frame = ip.crop(frame)
        should_break = run(frame, fn)
        if should_break:
            break

    cap.release()
    cv2.destroyAllWindows()


def extract_bounding_boxes_by_skin_threshold(image):
    skin = ip.extract_skin(image)

    binary_skin = ip.hsv_to_binary(skin)
    mask_binary = ip.find_largest_connected_component(binary_skin)

    # Find bounding boxes.
    bbox = ip.find_bounding_box_of_binary_img_with_single_component(mask_binary)
    square_bbox = ip.get_square_bbox(bbox, image)
    return skin, mask_binary, bbox, square_bbox


def run(img, fn):
    """
    Processes the frame and calls the `fn` with processed images as args.
    :param frame:
    :param fn:
    :return:
    """
    skin, binary_mask, bbox, sq_bbox = extract_bounding_boxes_by_skin_threshold(img)

    # Create image from the original with red/green boxes to show the boundary.
    frame_with_rect_bbox = ip.add_bounding_box_to_img(img, bbox)
    frame_with_rect_sq_bboxes = ip.add_bounding_box_to_img(frame_with_rect_bbox, sq_bbox, (0, 255, 0))

    # Crop frame and binary mask to the correct bounding box.
    hand = ip.crop_image_by_square_bbox(img, sq_bbox, size)
    hand_binary_mask = cv2.cvtColor(ip.crop_image_by_square_bbox(binary_mask, sq_bbox, size), cv2.COLOR_GRAY2RGB)

    # Do whatever with the preprocessed images.
    fn(img,
       frame_with_rect_sq_bboxes,
       skin,
       hand,
       binary_mask,
       hand_binary_mask,
       sq_bbox)


# def add_folder():
#     if not os.path.exists(path_to_captured_masks):
#         os.makedirs(path_to_captured_masks)
#
#     imgs = fetch_training_images_binary(path_to_captured_images, 700)
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