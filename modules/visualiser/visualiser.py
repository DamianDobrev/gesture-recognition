import cv2
import imutils
import numpy as np

from modules.image_processing.canvas import Canvas
from config import CONFIG


def draw_rectangle_in_center(img, color=(255, 255, 0)):
    """
    Appends a small rectangle in teh center of a BGR image.
    :param img: A BGR image with shape (X,Y,3).
    :param color: The color of the rectangle.
    :return: The image with added rectangle in the center.
    """
    height, width = img.shape[:2]
    img_rect = img.copy()
    cv2.rectangle(img_rect, (int(width / 2 - 3), int(height / 2 - 3)),
                  (int(width / 2 + 3), int(height / 2 + 3)), color, 1)
    return img_rect


def draw_bounding_box_in_img(img, bbox, color=(30, 0, 255), thresh=0):
    """
    Draws a bounding box (a rectangle) in an image.
    :param img: Any image with shape (X,Y,n).
    :param bbox: Bounding box rectangle in format [top_left_Y, top_left_X,
        bottom_right_Y, bottom_right_X].
    :param color: The color of the bbox.
    :param thresh: The threshold of the drawn bbox. Recommended to be 0.
    :return: The same image but with a drawn bounding box.
    """
    return cv2.rectangle(img.copy(), (bbox[1] - thresh, bbox[0] - thresh), (bbox[3] + thresh, bbox[2] + thresh), color, 2)


def visualise(img_conversions, texts_list, window_name='Gesture Recognition'):
    """
    Visualizes main UI consisting of:
    - left: original image, 3 processing methods.
    - right: information in form of texts passed as param.
    :param img_conversions: Image conversions as coming from converter.py.
    :param texts_list: A list of texts to be displayed.
    :param window_name: The name of the window, defaults to "Gesture Recognition".
    :return: Does not return anything.
    """
    orig_mark_center = draw_rectangle_in_center(img_conversions['orig_bboxes'])

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


def visualise_prediction_result(probabilities, classes, cox, coy, offset_padding):
    """
    Visualizes prediction result via bar-charts for the probabilities of each label
    and a simple coordinate system for the X and Y offsets.
    :param probabilities: A list of probabilities in percentage that add up to 100,
        e.g. [22, 58, 20].
    :param classes: The classes for these probabilities. Should be the same length
        as the `probabilities`.
    :param cox: Offset X from the center as ratio, in range 0..1.
    :param coy: Offset Y from the center as ratio, in range 0..1.
    :param offset_padding: How much space to give for offset to be drawn.
    :return: Does not return anything.
    """
    canv = Canvas((340, 900, 3))

    # Offset of prediction values.
    p_y_off = 40
    p_x_off = 200
    bar_width = 60
    bar_max_height = 250
    spacing = 10

    for idx, probability in enumerate(probabilities):
        x = spacing + bar_width * idx + spacing * idx + p_x_off
        color = (0, 50, 255) if probability > CONFIG['predicted_val_threshold'] else (0, 255, 255)
        height = int(bar_max_height * probability / 100)
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

    c_x = int(t_x_off + offset_padding * -cox + 50)
    c_y = int(t_y_off + offset_padding * coy + 50)
    cv2.circle(canv.print(), center=(c_x, c_y), radius=5, color=(200, 150, 250), thickness=-5)

    cv2.imshow('Predictions', canv.print())


def visualise_calibration(orig_mark_center, texts_list, window_name='Calibration'):
    """
    Displays a window that is suitable for calirbation. It can also be used for other things,
    hence this naming is not entirely correct. It differs from the `visualise` method
    because the images are displayed on top and one canvas is displayed beneath them,
    holding the needed texts.
    :param orig_mark_center: Images to be displayed on top, shapes (X,Y,1..3).
    :param texts_list: Lists of texts to be displayed in canvas.
    :param window_name: The name of the window, defaults to "Calibration".
    :return:
    """
    top_canvas = Canvas((400, orig_mark_center.shape[1], 3))

    line_num = 0

    for idx, text in enumerate(texts_list):
        top_canvas.draw_text(idx, text)
        line_num += 1

    stack1 = np.vstack([orig_mark_center, top_canvas.print()])

    cv2.imshow(window_name, stack1)
