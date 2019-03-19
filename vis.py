import cv2
import numpy as np

from image_processing.canvas import Canvas
from config import CONFIG

size = CONFIG['size']
window_name = 'Img with Bbox + processing.'


def append_rectangle_in_center(img, color=(255, 255, 0)):
    height, width = img.shape[:2]
    img_rect = img.copy()
    cv2.rectangle(img_rect, (int(width / 2 - 3), int(height / 2 - 3)),
                  (int(width / 2 + 3), int(height / 2 + 3)), color, 1)
    return img_rect


def visualise(params, orig, skin, binary_mask, hand, hand_binary_mask):
    orig_mark_center = append_rectangle_in_center(orig)

    vals_canvas = Canvas((size, 500, 3))
    vals_canvas.draw_text(1, 'hsv: ' + str(params['center_hsv']))
    vals_canvas.draw_text(2, 'res: ' + str(params['result']))

    empty_canvas = Canvas((size, 500, 3))

    stack1 = np.hstack([orig_mark_center, skin, vals_canvas.print()])
    stack2 = np.hstack([hand, hand_binary_mask, empty_canvas.print()])

    cv2.imshow(window_name, np.vstack([stack1, stack2]))
