import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import label
from skimage.measure import regionprops

bbthresh = 20


def to_50x50_monochrome(img):
    def to_monochrome(im):
        return np.array(Image.fromarray(im).convert('L'))

    img = cv2.resize(img, (50, 50))
    img = to_monochrome(img)
    img = np.array([img])  # shape should be like (1, 50, 50)
    return img


class ImageProcessor:
    def __init__(self, size, lower, upper):
        self.size = size
        self.lower = np.array(lower, dtype = "uint8")
        self.upper = np.array(upper, dtype = "uint8")

    def crop(self, img, size=None):
        if size is None:
            size = self.size
        h, w = img.shape[:2]
        h_sp = int((h - size) / 2)
        w_sp = int((w - size) / 2)
        frame = img[h_sp:h_sp+size, w_sp:w_sp+size]
        return frame

    def increase_saturation(self, img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[..., 1] = img_hsv[..., 1] * 1.4
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    def normalize_hist(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        # convert the YUV image back to BGR format
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def extract_skin(self, img):
        image = img.copy()
        frame_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        frame_hsv = cv2.GaussianBlur(frame_hsv, (3, 3), 1)
        # frame_hsv = cv2.medianBlur(frame_hsv, 3)
        frame_hsv = cv2.bilateralFilter(frame_hsv, 3, 45, 45)
        # frame_hsv = cv2.addWeighted(frame_hsv, 2, frame_hsv, -1, 0)

        skin_mask = cv2.inRange(frame_hsv, self.lower, self.upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # 2 erode, 2 dilate also works well.
        skin_mask = cv2.erode(skin_mask, kernel, iterations=3)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

        return cv2.bitwise_and(image, image, mask=skin_mask)

    def hsv_to_binary(self, image):
        binary_mask = cv2.inRange(image, np.array([5, 5, 5], dtype = "uint8"), np.array([179, 255, 255], dtype = "uint8"))
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        # another =  self.normalize_hist(image.copy())
        # cv2.imshow('asdsadsa', another)
        img = cv2.cvtColor(image.copy(), cv2.COLOR_HSV2BGR)
        binary_img = img[:,:,0] & binary_mask[:,:,0]

        def smooth_binary(img_in):
            img = img_in.copy()
            # Enable for daylight!
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            img = cv2.erode(img, kernel, iterations=3)
            img = cv2.dilate(img, kernel, iterations=2)
            return img

        def fill_holes(img_in):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            output = cv2.morphologyEx(img_in, cv2.MORPH_OPEN, kernel)
            return output

        def binary_fill_holes(img_in):
            im_out = ndimage.binary_fill_holes(img_in).astype(int)
            shit = img_in
            height, width = im_out.shape
            # do it in manual way because im_out has weird depth for some reason
            for y in range(0, height):
                for x in range(0, width):
                    shit[y, x] = 255 if im_out[y, x] >= 1 else 0
            return shit

        binary_img = smooth_binary(binary_img)
        binary_img = fill_holes(binary_img)
        binary_img = binary_fill_holes(binary_img)

        return binary_img

    def find_largest_connected_component(self, img):
        new_img = np.zeros_like(img)  # step 1
        for val in np.unique(img)[1:]:  # step 2
            mask = np.uint8(img == val)  # step 3
            labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  # step 4
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # step 5
            new_img[labels == largest_label] = val  # step 6
        return new_img

    def find_bounding_box_of_binary_img_with_single_component(self, binary_img, thresh=bbthresh):
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

    def add_bounding_box_to_img(self, img, bbox, color=(30, 0, 255), thresh=0):
        return cv2.rectangle(img.copy(), (bbox[1] - thresh, bbox[0] - thresh), (bbox[3] + thresh, bbox[2] + thresh), color, 2)

    def get_square_bbox(self, bbox, image, thresh=0):
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

    def crop_image_by_square_bbox(self, frame, square_bbox, size_width):
        if square_bbox[2]-square_bbox[0] != square_bbox[3]-square_bbox[1]:
            raise AttributeError('crop_image_by_square_bbox should only accept square bboxes')
        new_frame = frame[square_bbox[0]:square_bbox[2], square_bbox[1]:square_bbox[3]]
        height, width = new_frame.shape[:2]
        to_return = new_frame if width > 0 and height > 0 else frame
        return cv2.resize(to_return, (size_width, size_width))
