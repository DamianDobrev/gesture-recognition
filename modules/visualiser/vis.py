import cv2
import numpy as np

from modules.image_processing.canvas import Canvas
from config import CONFIG

# size = CONFIG['size']
window_name = CONFIG['imshow_window_name']

num_canvas_lines = 8


def append_rectangle_in_center(img, color=(255, 255, 0)):
    height, width = img.shape[:2]
    img_rect = img.copy()
    cv2.rectangle(img_rect, (int(width / 2 - 3), int(height / 2 - 3)),
                  (int(width / 2 + 3), int(height / 2 + 3)), color, 1)
    return img_rect


def visualise(img_conversions, texts_list):
    orig_mark_center = append_rectangle_in_center(img_conversions['orig_bboxes'])

    size = img_conversions['orig'].shape[0]
    top_canvas = Canvas((size, 500, 3))
    bot_canvas = Canvas((size, 500, 3))

    line_num = 0

    for idx, text in enumerate(texts_list):
        canvas = top_canvas if line_num <= num_canvas_lines else bot_canvas
        canvas.draw_text(idx, text)
        line_num += 1

    stack1 = np.hstack([orig_mark_center, img_conversions['skin'], top_canvas.print()])
    stack2 = np.hstack([img_conversions['hand_binary_mask'], img_conversions['skin_monochrome'], bot_canvas.print()])

    cv2.imshow(CONFIG['imshow_window_name'], np.vstack([stack1, stack2]))


def visualise_prediction(vals, classes):
    canv = Canvas((340, 800, 3))

    h_thresh = 40
    bar_width = 60
    bar_max_height = 250
    spacing = 10

    for idx, val in enumerate(vals):
        x = spacing + bar_width * idx + spacing * idx
        color = (0, 50, 255) if val > CONFIG['predicted_val_threshold'] else (0, 255, 255)
        height = int(bar_max_height * val / 100)
        top = bar_max_height + h_thresh - height
        cv2.rectangle(canv.print(), (x, top), (x + bar_width, bar_max_height + h_thresh), color, -1)
        cv2.putText(canv.print(),
                    classes[int(idx)],
                    (int(x), bar_max_height + h_thresh + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imshow('Predictions', canv.print())


def visualise_orig(orig_mark_center, texts_list):
    top_canvas = Canvas((400, orig_mark_center.shape[1], 3))

    line_num = 0

    for idx, text in enumerate(texts_list):
        top_canvas.draw_text(idx, text)
        line_num += 1

    stack1 = np.vstack([orig_mark_center, top_canvas.print()])

    cv2.imshow(CONFIG['imshow_window_name'], stack1)
