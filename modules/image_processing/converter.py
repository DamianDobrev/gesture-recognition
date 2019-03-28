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
        'center_hsv': get_center_hsv(img)
    }


def convert_img_for_test_or_prediction(ip, img):
    params = convert_image(ip, img)
    # new_img = params['orig_monochrome']
    new_img = params['skin_monochrome']
    # new_img = params['hand_binary_mask']
    new_img = resize_to_training_img_size(new_img)
    new_img = convert_to_one_channel_monochrome(new_img)
    # new_img = np.array([cv2.equalizeHist(new_img)])
    cv2.imshow('to_pred', new_img)
    return new_img, params


def convert_img_for_prediction(ip, img, image_processing_kind, image_size):
    img_conversions = convert_image(ip, img)
    new_img = img_conversions[image_processing_kind]
    new_img = resize_to_training_img_size(new_img, image_size)
    new_img = convert_to_one_channel_monochrome(new_img)
    cv2.imshow('Model sees this.', new_img)
    return new_img, img_conversions
