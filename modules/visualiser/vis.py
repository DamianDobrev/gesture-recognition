import cv2
import imutils
import numpy as np

from modules.image_processing.canvas import Canvas
from config import CONFIG


def append_rectangle_in_center(img, color=(255, 255, 0)):
    height, width = img.shape[:2]
    img_rect = img.copy()
    cv2.rectangle(img_rect, (int(width / 2 - 3), int(height / 2 - 3)),
                  (int(width / 2 + 3), int(height / 2 + 3)), color, 1)
    return img_rect



def append_bounding_box_to_img(img, bbox, color=(30, 0, 255), thresh=0):
    return cv2.rectangle(img.copy(), (bbox[1] - thresh, bbox[0] - thresh), (bbox[3] + thresh, bbox[2] + thresh), color, 2)


def visualise(img_conversions, texts_list, window_name='Gesture Recognition'):
    orig_mark_center = append_rectangle_in_center(img_conversions['orig_bboxes'])

    size = CONFIG['vis_size']
    top_canvas = Canvas((size, 600, 3))
    bot_canvas = Canvas((size, 600, 3))

    num_canvas_lines = 6

    for idx, text in enumerate(texts_list):
        canvas = top_canvas if idx < num_canvas_lines else bot_canvas
        canvas.draw_text(idx % num_canvas_lines, text)

    tl = imutils.resize(orig_mark_center, size, size)
    tr = imutils.resize(img_conversions['skin_orig'], size, size)
    bl = imutils.resize(img_conversions['hand_binary_mask'], size, size)
    br = imutils.resize(img_conversions['skin_monochrome'], size, size)

    stack1 = np.hstack([tl, tr, top_canvas.print()])
    stack2 = np.hstack([bl, br, bot_canvas.print()])

    cv2.imshow(window_name, np.vstack([stack1, stack2]))


def visualise_prediction(vals, classes, cox, coy, max_size):
    canv = Canvas((340, 900, 3))

    # Offset of prediction values.
    p_y_off = 40
    p_x_off = 200
    bar_width = 60
    bar_max_height = 250
    spacing = 10

    for idx, val in enumerate(vals):
        x = spacing + bar_width * idx + spacing * idx + p_x_off
        color = (0, 50, 255) if val > CONFIG['predicted_val_threshold'] else (0, 255, 255)
        height = int(bar_max_height * val / 100)
        top = bar_max_height + p_y_off - height
        cv2.rectangle(canv.print(), (x, top), (x + bar_width, bar_max_height + p_y_off), color, -1)
        cv2.putText(canv.print(),
                    classes[int(idx)],
                    (int(x), bar_max_height + p_y_off + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Offset of tracker.
    t_y_off = 100
    t_x_off = 50

    cv2.line(canv.print(), (t_x_off + 50, t_y_off + 0), (t_x_off + 50, t_y_off + 100), color=(150, 150, 250))
    cv2.line(canv.print(), (t_x_off + 0, t_y_off + 50), (t_x_off + 100, t_y_off + 50), color=(150, 150, 250))

    c_x = int(t_x_off + max_size * -cox + 50)
    c_y = int(t_y_off + max_size * coy + 50)
    cv2.circle(canv.print(), center=(c_x, c_y), radius=5, color=(200, 150, 250), thickness=-5)

    cv2.imshow('Predictions', canv.print())


def visualise_orig(orig_mark_center, texts_list, window_name='Gesture Recognition'):
    top_canvas = Canvas((400, orig_mark_center.shape[1], 3))

    line_num = 0

    for idx, text in enumerate(texts_list):
        top_canvas.draw_text(idx, text)
        line_num += 1

    stack1 = np.vstack([orig_mark_center, top_canvas.print()])

    cv2.imshow(window_name, stack1)
