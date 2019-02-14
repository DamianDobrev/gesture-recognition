import time

import cv2
from image_processor import extract_skin
import sys

print('Python version:  ' + sys.version)
print('Open CV version: ' + cv2. __version__)

# for i in range(1, 40):
#     img = cv2.imread('training_img/out' + str(i).zfill(3) + '.png')
#     skin, skin_mask, horizontal_concat = extract_skin(img)
#     cv2.imshow('skin00' + str(i).zfill(3), horizontal_concat)
#     # cv2.imshow('skin00' + str(i).zfill(3), skin_mask)

# img = cv2.imread('training_img/out001.png')
# skin, skin_mask, horizontal_concat = extract_skin(img)
# cv2.imshow('skin001', horizontal_concat)

# Camera!
camera = cv2.VideoCapture(0)
# cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

while True:
    (grabbed, frame) = camera.read()
    key = cv2.waitKey(1) & 0xff

    if not grabbed:
        break

    skin, skin_mask, horizontal_concat = extract_skin(frame)
    cv2.imshow('skin001', horizontal_concat)
    # cv2.waitKey(0)
    # time.sleep(0.01)
    # camera.release()
    # cv2.destroyAllWindows()

    # time.sleep(1)

print('shit')