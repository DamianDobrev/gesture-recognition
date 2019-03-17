import cv2
import numpy as np
from PIL import Image
from matplotlib.pyplot import cm

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
        skinMask = cv2.erode(skinMask, kernel, iterations=1)
        skinMask = cv2.dilate(skinMask, kernel, iterations=1)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        # skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

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
        return new_img
