import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

import numpy as np

style.use('fivethirtyeight')

class ValuesAnimator:
    def show_chart(self, values):
        chart = np.full(
            shape=200,
            fill_value=255,
            dtype=np.int)

        cv2.imshow('kur', )