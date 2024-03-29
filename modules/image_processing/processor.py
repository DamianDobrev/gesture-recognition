import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import label
from skimage.measure import regionprops

from config import CONFIG


def convert_to_one_channel_monochrome(img):
    """
    Converts a BGR image to a single channel monochrome image.
    :param img: BGR image, with shape (X,Y,3).
    :return: The image in monochrome, with shape (X,Y,1).
    """
    img_new = np.array(Image.fromarray(img).convert('L'))
    img_new = np.array([img_new])  # shape should be like (1, 50, 50)
    img_new = np.moveaxis(img_new, 0, -1)  # shape should be like (50, 50, 1)
    return img_new


def extract_skin(img, l_hsv_bound, h_hsv_bound):
    """
    Accepts BGR image and extracts skin based on the lower and upper value from params.
    :param img: A BGR image. Shape will be of type (X,Y,3).
    :param h_hsv_bound: An array, list or np array with 3 values, [hue, sat, val].
    :param l_hsv_bound: An array, list or np array with 3 values, [hue, sat, val].
    :return: Returns 2 elements
        - A BGR image with shape (X,Y,3). All none-skin pixels are zeros.
        - A binary image with shape (X,Y).
    """
    image_copy = img.copy()
    frame_hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
    frame_hsv = cv2.GaussianBlur(frame_hsv, (5, 5), 1)
    frame_hsv = cv2.bilateralFilter(frame_hsv, 3, 45, 45)

    skin_mask = cv2.inRange(frame_hsv, np.array(l_hsv_bound, dtype=np.int32), np.array(h_hsv_bound, dtype=np.int32))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=3)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

    h, w, _ = img.shape
    binary_mask = np.array(skin_mask).reshape((h, w))
    skin_separated = cv2.bitwise_and(image_copy, image_copy, mask=skin_mask)

    return skin_separated, binary_mask


def crop_from_center(img, size):
    """
    Takes a BGR (or even any other) image and size and crops the
    centered square of the image.
    :param img: BGR (or even any other) image.
    :param size: The size of the square that will be cropped.
    :return:
    """
    h, w = img.shape[:2]
    h_sp = int((h - size) / 2)
    w_sp = int((w - size) / 2)
    frame = img[h_sp:h_sp+size, w_sp:w_sp+size]
    return frame


def fill_and_smooth_binary_mask(binary_mask):
    """
    Smooths a binary mask and fills all its holes.
    :param binary_mask: A binary mask of shape (X,Y).
    :return: Smoothened binary mask without holes of shape (X,Y).
    """

    def smooth_binary(mask):
        mask_in = mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.dilate(mask_in, kernel, iterations=3)
        img = cv2.erode(img, kernel, iterations=2)
        img = np.array(img)
        return np.bitwise_or(mask_in, img)

    def fill_holes_binary(img_in):
        im_out = ndimage.binary_fill_holes(img_in).astype(int)
        return cv2.inRange(im_out, np.array([1]), np.array([255]))

    binary_img = smooth_binary(binary_mask)
    binary_img = fill_holes_binary(binary_img)

    return binary_img


def isolate_largest_connected_component(binary_mask):
    """
    Finds the largest connected component in a binary image and
    isolates it, meaning that only this component is left in the
    frame as-it-is, everything else is set to 0.
    :param binary_mask: A binary mask of shape (X,Y).
    :return: A binary mask of shape (X,Y) with only 1 connected
        component.
    """

    new_img = np.zeros_like(binary_mask)
    for val in np.unique(binary_mask)[1:]:
        second_mask = np.uint8(binary_mask == val)
        labels, stats = cv2.connectedComponentsWithStats(second_mask, 4)[1:3]
        biggest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        new_img[labels == biggest_label] = val
    return new_img


def find_bounding_box_of_binary_img_with_single_component(binary_img, thresh=CONFIG['bbox_threshold']):
    """
    Finds the bounding box of the connected component inside a
    binary image and returns it.
    :param binary_img: A binary image of shape (X,Y).
    :param thresh: A threshold working as a padding to be applied.
    :return: An array of 4 values:
        - [0]: The Y coordinate of the top-left point.
        - [1]: The X coordinate of the top-left point.
        - [2]: The Y coordinate of the bottom-right point.
        - [3]: The X coordinate of the bottom-right point.
    In the case such component does not exist, or is very small,
    then [0,0,0,0] is returned.
    """

    mask_label = label(binary_img)
    props = regionprops(mask_label)
    height, width = binary_img.shape
    if len(props) > 0:
        # We only need the first one since it's a single component.
        bbox = props[0].bbox
        # Math and min are because by adding/subtracting a threshold, the values may go beyond the image. We
        # want the image to contain the bounding box.
        bbox = [max(bbox[0] - thresh, 0), max(bbox[1] - thresh, 0), min(bbox[2] + thresh, height), min(bbox[3] + thresh, width)]
        return bbox
    return [0, 0, 0, 0]  # Default, if the mask is full of zeros we need to return something.


def get_square_bbox(rect_bbox, image, thresh=0):
    """
    Takes an image and a rectangular bbox, and returns a squared bbox which "envelops" the rectangular bbox.
    Threshold can be provided as well. TODO test the threshold.
    :param image: an image. Can be 1 or 3 dimensions.
    :param bbox: in format [top_left_Y, top_left_X, bottom_right_Y, bottom_right_X]
    :return:
    """
    image_height, image_width = image.shape if image.ndim == 1 else image.shape[:2]

    new_top_left_x = rect_bbox[1] - thresh
    new_bottom_right_x = rect_bbox[3] + thresh

    new_top_left_y = rect_bbox[0] - thresh
    new_bottom_right_y = rect_bbox[2] + thresh

    new_bbox_width = new_bottom_right_x - new_top_left_x
    new_bbox_height = new_bottom_right_y - new_top_left_y

    def crop_axis(lower_bbox_val, higher_bbox_val, frame_size_on_axis, diff, new_size):
        lower_new_val = lower_bbox_val - diff
        higher_new_val = higher_bbox_val + diff

        # Calc new top left x and new bottom left x
        if lower_new_val < 0:
            higher_new_val = new_size
            lower_new_val = 0
        elif higher_new_val > frame_size_on_axis:
            lower_new_val = frame_size_on_axis - new_size
            higher_new_val = frame_size_on_axis

        return lower_new_val, higher_new_val

    if new_bbox_height > new_bbox_width:
        px_to_add_to_each_side = (new_bbox_height - new_bbox_width) / 2
        new_top_left_x, new_bottom_right_x = crop_axis(new_top_left_x, new_bottom_right_x, image_width, px_to_add_to_each_side, new_bbox_height)
    else:
        px_to_add_to_each_side = (new_bbox_width - new_bbox_height) / 2
        new_top_left_y, new_bottom_right_y = crop_axis(new_top_left_y, new_bottom_right_y, image_height, px_to_add_to_each_side, new_bbox_width)

    # This assures a proper error will be thrown if something is sketchy.
    if new_bottom_right_y > image_height:
        raise ArithmeticError('Bottom edge should not be higher than frame height.')
    elif new_top_left_y < 0:
        raise ArithmeticError('Top edge should not be less than 0.')
    elif new_bottom_right_x > image_width:
        raise ArithmeticError('Right edge should not be higher than frame width.')
    elif new_top_left_x < 0:
        raise ArithmeticError('Left edge should not be less than 0.')

    return [int(new_top_left_y), int(new_top_left_x), int(new_bottom_right_y), int(new_bottom_right_x)]


def crop_image_by_square_bbox(img, square_bbox, size_width):
    """
    By given an image and a square bbox, it crops the image around the bbox.
    If the bbox is not square, an AttributeError is thrown.
    :param img: The image to crop with shape (X,Y,3).
    :param square_bbox: The square bouunding box.
    :param size_width: The size to which crop.
    :return: Am image of size (size_width, size_width).
    """
    if square_bbox[2]-square_bbox[0] != square_bbox[3]-square_bbox[1]:
        raise AttributeError('crop_image_by_square_bbox should only accept square bboxes')
    new_frame = img[square_bbox[0]:square_bbox[2], square_bbox[1]:square_bbox[3]]
    height, width = new_frame.shape[:2]
    to_return = new_frame if width > 0 and height > 0 else img
    return cv2.resize(to_return, (size_width, size_width))
