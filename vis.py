import cv2
import numpy as np

from image_processing.canvas import Canvas
from config import CONFIG

size = CONFIG['size']
window_name = 'Img with Bbox + processing.'

num_canvas_lines = 8


def append_rectangle_in_center(img, color=(255, 255, 0)):
    height, width = img.shape[:2]
    img_rect = img.copy()
    cv2.rectangle(img_rect, (int(width / 2 - 3), int(height / 2 - 3)),
                  (int(width / 2 + 3), int(height / 2 + 3)), color, 1)
    return img_rect


def visualise(img_conversions, texts_list):
    orig_mark_center = append_rectangle_in_center(img_conversions['orig_bboxes'])

    top_canvas = Canvas((size, 500, 3))
    bot_canvas = Canvas((size, 500, 3))

    line_num = 0

    for idx, text in enumerate(texts_list):
        canvas = top_canvas if line_num <= num_canvas_lines else bot_canvas
        canvas.draw_text(idx, text)
        line_num += 1

    stack1 = np.hstack([orig_mark_center, img_conversions['skin'], top_canvas.print()])
    stack2 = np.hstack([img_conversions['hand'], img_conversions['hand_binary_mask'], bot_canvas.print()])

    cv2.imshow(window_name, np.vstack([stack1, stack2]))


def visualise_orig(orig_mark_center, texts_list):
    top_canvas = Canvas((400, orig_mark_center.shape[1], 3))

    line_num = 0

    for idx, text in enumerate(texts_list):
        top_canvas.draw_text(idx, text)
        line_num += 1

    stack1 = np.vstack([orig_mark_center, top_canvas.print()])

    cv2.imshow(window_name, stack1)
