import cv2
import imutils

from config import CONFIG
import modules.image_processing.processor as imp

size = CONFIG['size']


def loop(fn):
    cap = cv2.VideoCapture(0)
    # This is supposed to prevent the camera from over/underexponsing randomly.
    cap.set(15, 0.00001)

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, height=size)
        frame = imp.crop(frame, size)
        should_break = fn(frame)
        if should_break:
            break

    cap.release()
    cv2.destroyAllWindows()
