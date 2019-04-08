import random

import cv2
import numpy as np
from PIL import Image

from config import CONFIG
import modules.image_processing.processor as imp
from modules.visualiser.vis import draw_bounding_box_in_img

size = CONFIG['size']


def get_center_hsv(img):
    """
    Converts an image to HSV and returns the hsv value of its center pixel.
    :param img: A BGR image with shape (X,Y,3).
    :return: HSV value of the center pixel [h,s,v].
    """
    h, w = img.shape[:2]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.GaussianBlur(img_hsv, (3, 3), 0)
    center_px_hsv = img_hsv[int(h / 2), int(w / 2)]
    return np.array(center_px_hsv)


def extract_bounding_boxes_by_skin_threshold(image, l_hsv_thresh, u_hsv_thresh):
    """
    This method encapsulates the calls to few other methods that perform:
    - skin colored pixels extraction
    - mask of the skin colored pixels only containing the largest connected component (LCC)
    - rectangular bounding box around the LCC
    - square bounding box around the LCC
    :param image: A BGR image with shape (X,Y,3).
    :param l_hsv_thresh: Lower skin color threshold of type [h,s,v].
    :param u_hsv_thresh: Upper skin color threshold of type [h,s,v].
    :return:
    """
    skin, mask_binary = imp.extract_skin(image, l_hsv_thresh, u_hsv_thresh)

    mask_binary = imp.fill_and_smooth_binary_mask(mask_binary)
    mask_binary = imp.isolate_largest_connected_component(mask_binary)

    # Find bounding boxes.
    bbox = imp.find_bounding_box_of_binary_img_with_single_component(mask_binary)
    square_bbox = imp.get_square_bbox(bbox, image)
    return skin, mask_binary, bbox, square_bbox


def to_monochrome(im):
    """
    Transforms an image to monochrome.
    :param im: Image with shape (X,Y,3).
    :return: Image with shape (X,Y).
    """
    return np.array(Image.fromarray(im).convert('L'))


def convert_image(img, l_hsv_thresh, u_hsv_thresh):
    """
    Applies processing techniques to an image and returns some of its derivations.
    :param img:
    :param l_hsv_thresh: The lower HSV boundary of skin color.
    :param u_hsv_thresh: The upper HSV boundary of skin color.
    :return:
    """
    skin, binary_mask, bbox, sq_bbox = extract_bounding_boxes_by_skin_threshold(img, l_hsv_thresh, u_hsv_thresh)
    center_offset_y = (sq_bbox[2] - sq_bbox[0]) / 2 + sq_bbox[0] - CONFIG['size'] / 2
    center_offset_x = (sq_bbox[3] - sq_bbox[1]) / 2 + sq_bbox[1] - CONFIG['size'] / 2

    def in_perc(val):
        val = min(45, val)
        val = max(-45, val)
        return int(val) / 45

    center_offset_x = in_perc(center_offset_x)
    center_offset_y = in_perc(center_offset_y)

    # Create image from the original with red/green boxes to show the boundary.
    frame_with_rect_bbox = draw_bounding_box_in_img(img, bbox)
    frame_with_rect_sq_bboxes = draw_bounding_box_in_img(frame_with_rect_bbox, sq_bbox, (0, 255, 0))

    # Crop frame and binary mask to the correct bounding box.
    hand = imp.crop_image_by_square_bbox(img, sq_bbox, size)
    skin_orig = skin.copy()
    skin = cv2.bitwise_and(img, img, mask=binary_mask)
    skin = imp.crop_image_by_square_bbox(skin, sq_bbox, size)
    hand_binary_mask = imp.crop_image_by_square_bbox(binary_mask, sq_bbox, size)
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


def convert_img_for_prediction(img, l_hsv_thresh, u_hsv_thresh, image_processing_kind, image_size):
    img_conversions = convert_image(img, l_hsv_thresh, u_hsv_thresh)
    new_img = img_conversions[image_processing_kind]
    new_img = cv2.resize(new_img, (image_size, image_size))
    new_img = imp.convert_to_one_channel_monochrome(new_img)
    # Uncomment this to visualise what the model sees.
    # cv2.imshow('Model sees this.', new_img)
    return new_img, img_conversions


def randomly_rotate_and_change_alpha(im, cw=False):
    """
    Takes an image and randomly applies a rotation and a contrast change.
    :param im: An image with shape (X,Y,1).
    :param cw: If true, rotation will be performed clockwise, otherwise it
        will be performed counter-clockwise.
    :return: An image with shape (X,Y,1).
    """
    lower, upper = (10, 30) if cw else (-30, -10)
    rand = random.randint(lower, upper)
    m = cv2.getRotationMatrix2D((100, 100), rand, 1)
    rot = cv2.warpAffine(im, m, (CONFIG['size'], CONFIG['size']))
    alpha = random.uniform(0.6, 1.4)
    out = cv2.addWeighted(rot, alpha, rot, 0, 0)
    return out


def randomly_hide_parts(im):
    """
    Adds a random rectangle somewhere around the middle of the image to
    hide features.
    :param im: An image with shape (X,Y,1).
    :return: An image with shape (X,Y,1). There is 75% chance this image
        is th original image. 25% there is a random rectangle.
    """
    # There is 75% chance we don't do anything.
    if random.randint(1, 4) is not 2:
        return im

    s = CONFIG['size']
    w = random.randint(s/10, s/4)
    h = random.randint(s/10, s/4)
    x = random.randint(s/3, s/2)
    y = random.randint(s/3, s/2)

    im = cv2.rectangle(im, (y, x), (y+h, x+w), (0, 0, 0), thickness=-1)
    return im


def augment_image(img, num_augm_imgs=CONFIG['augmentation_count']):
    """
    Takes an image and performs random augmentation techniques in order to
    change its rotation and/or contrast, and to add a black "obstacle"
    at random place on the image, thus hiding random features.
    :param img: A grayscale image with shape (X,Y,1).
    :param num_augm_imgs: The number of images to be added.
    :return: A list of images, where the first image is the original, and
        the number of other images depends on `num_augm_imgs`. The shapes
        will be the same as the shape of the input image - (X,Y,1).
    """

    augmented_imgs = [img]
    for index in range(num_augm_imgs):
        new_im = randomly_rotate_and_change_alpha(img, index % 2 is 0)
        new_im = randomly_hide_parts(new_im)
        augmented_imgs.append(new_im)

    return augmented_imgs
