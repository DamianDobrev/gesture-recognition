import numpy as np
import cv2
import imutils


def crop(img):
    frame = imutils.resize(img, width=400)
    frame = frame[0:200, 100:300]
    frame = to_gray(frame)
    return frame


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def extract_skin (img):
    # No fucking clue how that works
    lower = np.array([0, 35, 70], dtype="uint8")
    upper = np.array([255, 180, 180], dtype="uint8")

    frame = crop(img)

    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    _skin_mask = cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    _skin_mask = cv2.erode(_skin_mask, kernel, iterations = 1)
    _skin_mask = cv2.dilate(_skin_mask, kernel, iterations = 1)

    _skin_mask = cv2.GaussianBlur(_skin_mask, (3, 3), 0)
    # This extracts the biggest skin object in the pic.
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
    # new_skin_mask = cv2.GaussianBlur(new_skin_mask, (7, 7), 0)

    skin = cv2.bitwise_and(frame, frame, mask=new_skin_mask)

    horizontal_concat = np.concatenate((frame, skin), axis=1)
    return skin, new_skin_mask, horizontal_concat, frame

def build_img():
    for i in range(1, 40):
        img = cv2.imread('training_img/out' + str(i).zfill(3) + '.png')
        # skin, skin_mask, horizontal_concat, frame = extract_skin(img)
        frame = crop(img)
        cv2.imshow('skin00' + str(i).zfill(3), frame)
        cv2.imwrite('training_img_crop/skin' + str(i).zfill(3) + '.png', frame)


# img = cv2.imread('training_img_crop/skin001.png')
#
# cv2.imshow('no_processing', img)
# cv2.imshow('processing', crop(img))
# crop(img)

# print('kur')
# build_img()



# cv2.waitKey(0)
