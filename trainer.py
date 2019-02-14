import numpy as np
import cv2
import keras
import imutils


def extract_skin (img):
    # No fucking clue how that works
    lower = np.array([0, 35, 70], dtype="uint8")
    upper = np.array([255, 180, 180], dtype="uint8")

    frame = imutils.resize(img, width=400)
    frame = frame[0:200, 100:300]

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    _skin_mask = cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    _skin_mask = cv2.erode(_skin_mask, kernel, iterations = 1)
    _skin_mask = cv2.dilate(_skin_mask, kernel, iterations = 1)

    _skin_mask = cv2.GaussianBlur(_skin_mask, (3, 3), 0)
    # new_skin_mask = np.zeros_like(_skin_mask)                                        # step 1
    # for val in np.unique(_skin_mask)[1:]:                                      # step 2
    #     mask = np.uint8(_skin_mask == val)                                     # step 3
    #     labels, stats = cv2.connectedComponentsWithStats(mask, 4)[1:3]  # step 4
    #     largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])      # step 5
    #     new_skin_mask[labels == largest_label] = val
    new_skin_mask = _skin_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    new_skin_mask = cv2.erode(new_skin_mask, kernel, iterations = 1)
    new_skin_mask = cv2.dilate(new_skin_mask, kernel, iterations = 1)
    new_skin_mask = cv2.GaussianBlur(new_skin_mask, (7, 7), 0)

    skin = cv2.bitwise_and(frame, frame, mask=new_skin_mask)

    horizontal_concat = np.concatenate((frame, skin), axis=1)
    return skin, new_skin_mask, horizontal_concat
