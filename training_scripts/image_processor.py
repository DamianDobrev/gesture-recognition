import cv2
import numpy as np
from PIL import Image
from matplotlib.pyplot import cm
from skimage.morphology import label
from skimage.measure import regionprops

bbthresh = 10

class ImageProcessor:
    def __init__(self, lower, upper):
        self.lower = np.array(lower, dtype = "uint8")
        self.upper = np.array(upper, dtype = "uint8")


    def preprocess_frame(self, image):
        # TODO Add preprocessing here.
        # cv2.imshow('blah', image)

        frame = image
        # resize the frame, convert it to the HSV color space,
        # and determine the HSV pixel intensities that fall into
        # the speicifed upper and lower boundaries
        converted = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        skinMask = cv2.inRange(converted, self.lower, self.upper)

        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (1, 1), 0)

        skin = cv2.bitwise_and(frame, frame, mask=skinMask)

        # show the skin in the image along with the mask
        return skin

    def in_binary(self, image):
        new_img = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY)
        height, width = new_img.shape[:2]
        for y in range(0, height):
            for x in range(0, width):
                # threshold the pixel
                new_img[y, x] = 255 if new_img[y, x] >= 10 else 0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        new_img = cv2.morphologyEx(new_img, cv2.MORPH_OPEN, kernel)

        return new_img

    def find_largest_connected_component(self, img):
        new_img = np.zeros_like(img)  # step 1
        for val in np.unique(img)[1:]:  # step 2
            mask = np.uint8(img == val)  # step 3
            labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  # step 4
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # step 5
            new_img[labels == largest_label] = val  # step 6

        print(new_img)
        return new_img

    def find_bounding_box_of_single_component(self, mask_binary):
        mask_label = label(mask_binary)
        props = regionprops(mask_label)
        if len(props) > 0:
            # We only need the first one since it's a single component.
            bbox = props[0].bbox
            return bbox
        # height, width = mask_binary.shape
        return [0, 0, 0, 0] # Default, if the mask is full of zeros we need to return something.

    def add_bounding_box_to_img(self, img, bbox, color=(30, 0, 255)):
        return cv2.rectangle(img.copy(), (bbox[1] - bbthresh, bbox[0] - bbthresh), (bbox[3] + bbthresh, bbox[2] + bbthresh), color, 2)

    def get_square_bbox(self, bbox, frame_width, frame_height):
        """
        Returns square new image .
        :param frame:
        :param bbox: in format [top_left_Y, top_left_X, bottom_right_Y, bottom_right_X]
        :return:
        """

        width = bbox[3] - bbox[1]
        height = bbox[2] - bbox[0]

        new_top_left_x = bbox[1]
        new_bottom_right_x = bbox[3]

        if height >= width:
            diff = (height - width) / 2

            new_top_left_x = bbox[1] - diff
            new_bottom_right_x = bbox[3] + diff

            # Calc new top left x and new bottom left x
            if new_top_left_x < 0:
                new_bottom_right_x = bbox[3] - bbox[1] + diff*2
                new_top_left_x = 0
            elif new_bottom_right_x > frame_width:
                new_top_left_x = frame_width - (bbox[3] - bbox[1]) - diff*2
                new_bottom_right_x = frame_width


        return [bbox[0], int(new_top_left_x), bbox[2], int(new_bottom_right_x)]

