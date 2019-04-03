import cv2
import numpy as np


class Canvas:
    def __init__(self, size):
        self.canvas = np.zeros(size, np.uint8)

    def __str__(self):
        return self.canvas

    def draw_text(self, line_num=1, text=''):
        bottom_left_corner = (20, 20 + line_num * 20)
        font_scale = 0.4
        font_color = (255, 255, 255)
        line_type = 1
        cv2.putText(self.canvas, text,
                    bottom_left_corner,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, font_color, line_type)

    def print(self):
        return self.canvas

