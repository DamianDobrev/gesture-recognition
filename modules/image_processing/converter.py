import random

import cv2
import numpy as np
from PIL import Image

from config import CONFIG
from modules.image_processing.processor import convert_to_one_channel_monochrome, \
    resize_to_training_img_size

size = CONFIG['size']


def get_center_hsv(img):
    h, w = img.shape[:2]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.GaussianBlur(img_hsv, (3, 3), 0)
    center_px_hsv = img_hsv[int(h / 2), int(w / 2)]
    return center_px_hsv


def extract_bounding_boxes_by_skin_threshold(ip, image):
    skin = ip.extract_skin(image)

    binary_skin = ip.hsv_to_binary(skin)

    mask_binary = ip.find_largest_connected_component(binary_skin)

    # Find bounding boxes.
    bbox = ip.find_bounding_box_of_binary_img_with_single_component(mask_binary)
    square_bbox = ip.get_square_bbox(bbox, image)
    return skin, mask_binary, bbox, square_bbox


def to_monochrome(im):
    return np.array(Image.fromarray(im).convert('L'))


def convert_image(ip, img):
    skin, binary_mask, bbox, sq_bbox = extract_bounding_boxes_by_skin_threshold(ip, img)
    center_offset_y = (sq_bbox[2] - sq_bbox[0]) / 2 + sq_bbox[0] - CONFIG['size'] / 2
    center_offset_x = (sq_bbox[3] - sq_bbox[1]) / 2 + sq_bbox[1] - CONFIG['size'] / 2

    # Create image from the original with red/green boxes to show the boundary.
    frame_with_rect_bbox = ip.add_bounding_box_to_img(img, bbox)
    frame_with_rect_sq_bboxes = ip.add_bounding_box_to_img(frame_with_rect_bbox, sq_bbox, (0, 255, 0))

    # Crop frame and binary mask to the correct bounding box.
    hand = ip.crop_image_by_square_bbox(img, sq_bbox, size)
    skin_orig = skin.copy()
    skin = cv2.bitwise_and(skin, skin, mask=binary_mask)
    skin = ip.crop_image_by_square_bbox(skin, sq_bbox, size)
    hand_binary_mask = ip.crop_image_by_square_bbox(binary_mask, sq_bbox, size)
    hand_binary_mask = cv2.cvtColor(hand_binary_mask, cv2.COLOR_GRAY2BGR)

    skin_monochrome = cv2.cvtColor(cv2.equalizeHist(to_monochrome(skin)), cv2.COLOR_GRAY2BGR)
    orig_monochrome = cv2.cvtColor(cv2.equalizeHist(to_monochrome(img)), cv2.COLOR_GRAY2BGR)
    return {
        'orig': img,
        'orig_monochrome': orig_monochrome,
        'orig_bboxes': frame_with_rect_sq_bboxes,
        'skin': skin,
        'skin_orig': skin_orig,
        'skin_monochrome': skin_monochrome,
        'hand': hand,
        'binary_mask': cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR),
        'hand_binary_mask': hand_binary_mask,
        'sq_bbox': sq_bbox,
        'center_hsv': get_center_hsv(img),
        'center_offset_x': center_offset_x,
        'center_offset_y': center_offset_y
    }


def convert_img_for_prediction(ip, img, image_processing_kind, image_size):
    img_conversions = convert_image(ip, img)
    new_img = img_conversions[image_processing_kind]
    new_img = resize_to_training_img_size(new_img, image_size)
    new_img = convert_to_one_channel_monochrome(new_img)
    # Uncomment this to visualise what the model sees.
    # cv2.imshow('Model sees this.', new_img)
    return new_img, img_conversions


def randomly_rotate_and_change_alpha(im, cw=False):
    lower, upper = (10, 30) if cw else (-30, -10)
    rand = random.randint(lower, upper)
    m = cv2.getRotationMatrix2D((100, 100), rand, 1)
    rot = cv2.warpAffine(im, m, (CONFIG['size'], CONFIG['size']))
    alpha = random.uniform(0.4, 1.6)
    out = cv2.addWeighted(rot, alpha, rot, 0, 0)
    return out


def randomly_hide_parts(im):
    # There is 75% chance we don't do anything.
    if random.randint(1, 4) is not 2:
        return im

    w = random.randint(20, 50)
    h = random.randint(20, 50)
    x = random.randint(70, 130)
    y = random.randint(70, 130)

    im = cv2.rectangle(im, (y, x), (y+h, x+w), (0, 0, 0), thickness=-1)
    return im


def augment_image(img):
    augmented_imgs = []
    for index in range(4):
        new_im = randomly_rotate_and_change_alpha(img, index % 2 is 0)
        new_im = randomly_hide_parts(new_im)
        augmented_imgs.append(new_im)

    return augmented_imgs


# augimg = cv2.imread('/Users/damian/Desktop/indivpr/__training_data/min_1200_per_class/skin_monochrome_augmented/10/0007.png')
# cv2.imshow('ORIGINAL', augimg)
# new = augment_image(augimg)
# for idx, im in enumerate(new):
#     cv2.imshow(str(idx), im)
#
# cv2.waitKey(0)