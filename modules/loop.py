import cv2
import imutils

from config import CONFIG
import modules.image_processing.processor as imp

size = CONFIG['size']


def loop_camera_frames(fn):
    """
    Starts the web camera, takes snapshots of each frame and
    calls the passed function with the frame. It stops the camera
    when the function returns truthy value.
    :param fn: A function to be called each frame. Params:
        - the frame as BGR image with shape (X,Y,3).
    :return: Does not return anything.
    """
    cap = cv2.VideoCapture(0)
    # This is supposed to prevent the camera from over/underexponsing randomly.
    cap.set(15, 0.00001)

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, height=size)
        frame = imp.crop_from_center(frame, size)
        should_break = fn(frame)
        if should_break:
            break

    cap.release()
    cv2.destroyAllWindows()
