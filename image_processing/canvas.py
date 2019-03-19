import cv2
import numpy as np


class Canvas:
    def __init__(self, size):
        self.canvas = np.zeros(size, np.uint8)

    def __str__(self):
        return self.canvas

    def draw_text(self, line_num=1, text=''):
        bottomLeftCornerOfText = (20, 50 + line_num * 20)
        fontScale = 0.4
        fontColor = (255, 255, 255)
        lineType = 1
        cv2.putText(self.canvas, text,
                    bottomLeftCornerOfText,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale,
                    fontColor,
                    lineType)

    def print(self):
        return self.canvas

