import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import label
from skimage.measure import regionprops

from config import CONFIG


def convert_to_one_channel_monochrome(img):
    img_new = np.array(Image.fromarray(img).convert('L'))
    img_new = np.array([img_new])  # shape should be like (1, 50, 50)
    img_new = np.moveaxis(img_new, 0, -1)  # shape should be like (50, 50, 1)
    return img_new


def resize_to_training_img_size(img, size=CONFIG['training_img_size']):
    return cv2.resize(img, (size, size))


class Processor:
    def __init__(self, size, lower, upper):
        self.size = size
        self.lower = np.array(lower, dtype="uint8")
        self.upper = np.array(upper, dtype="uint8")

    def extract_skin(self, img):
        """
        Accepts BGR image and extracts skin based on the lower and upper value of the processor.
        :param img:
        :return: A BGR image. All none-skin pixels are zeros.
        """
        image_copy = img.copy()
        frame_hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV)
        frame_hsv = cv2.GaussianBlur(frame_hsv, (5, 5), 1)
        frame_hsv = cv2.bilateralFilter(frame_hsv, 3, 45, 45)

        skin_mask = cv2.inRange(frame_hsv, self.lower, self.upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.erode(skin_mask, kernel, iterations=3)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

        h, w, _ = img.shape
        binary_mask = np.array(skin_mask).reshape((h, w))
        skin_separated = cv2.bitwise_and(image_copy, image_copy, mask=skin_mask)

        return skin_separated, binary_mask

    @staticmethod
    def crop(img, size):
        h, w = img.shape[:2]
        h_sp = int((h - size) / 2)
        w_sp = int((w - size) / 2)
        frame = img[h_sp:h_sp+size, w_sp:w_sp+size]
        return frame

    @staticmethod
    def fill_and_smooth_binary_mask(binary_mask):
        """
        Smooths a binary mask and fills all its holes.
        :param binary_mask: A binary mask of shape (X,X).
        :return: Smoothened binary mask without holes of shape (X,X).
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

    @staticmethod
    def find_largest_connected_component(img):
        new_img = np.zeros_like(img)  # step 1
        for val in np.unique(img)[1:]:  # step 2
            mask = np.uint8(img == val)  # step 3
            labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  # step 4
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # step 5
            new_img[labels == largest_label] = val  # step 6
        return new_img

    @staticmethod
    def find_bounding_box_of_binary_img_with_single_component(binary_img, thresh=CONFIG['bbox_threshold']):
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

    @staticmethod
    def add_bounding_box_to_img(img, bbox, color=(30, 0, 255), thresh=0):
        return cv2.rectangle(img.copy(), (bbox[1] - thresh, bbox[0] - thresh), (bbox[3] + thresh, bbox[2] + thresh), color, 2)

    @staticmethod
    def get_square_bbox(bbox, image, thresh=0):
        """
        Takes an image and a rectangular bbox, and returns a squared bbox which "envelops" the rectangular bbox.
        Threshold can be provided as well. TODO test the threshold.
        :param image: an image. Can be 1 or 3 dimensions.
        :param bbox: in format [top_left_Y, top_left_X, bottom_right_Y, bottom_right_X]
        :return:
        """
        image_height, image_width = image.shape if image.ndim == 1 else image.shape[:2]

        new_top_left_x = bbox[1] - thresh
        new_bottom_right_x = bbox[3] + thresh

        new_top_left_y = bbox[0] - thresh
        new_bottom_right_y = bbox[2] + thresh

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

        # These are important assertions because I get weird values.
        if new_bottom_right_y > image_height:
            raise ArithmeticError('Bottom edge should not be higher than frame height.')
        elif new_top_left_y < 0:
            raise ArithmeticError('Top edge should not be less than 0.')
        elif new_bottom_right_x > image_width:
            raise ArithmeticError('Right edge should not be higher than frame width.')
        elif new_top_left_x < 0:
            raise ArithmeticError('Left edge should not be less than 0.')

        return [int(new_top_left_y), int(new_top_left_x), int(new_bottom_right_y), int(new_bottom_right_x)]

    @staticmethod
    def crop_image_by_square_bbox(frame, square_bbox, size_width):
        if square_bbox[2]-square_bbox[0] != square_bbox[3]-square_bbox[1]:
            raise AttributeError('crop_image_by_square_bbox should only accept square bboxes')
        new_frame = frame[square_bbox[0]:square_bbox[2], square_bbox[1]:square_bbox[3]]
        height, width = new_frame.shape[:2]
        to_return = new_frame if width > 0 and height > 0 else frame
        return cv2.resize(to_return, (size_width, size_width))
