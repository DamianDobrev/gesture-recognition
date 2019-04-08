import unittest

import cv2
import numpy as np

from modules.image_processing.processor import isolate_largest_connected_component, get_square_bbox, \
    crop_image_by_square_bbox
from modules.model.model import split_data


class TestImageMethods(unittest.TestCase):
    def test_isolate_largest_connected_component(self):
        """
        When given a binary array, isolate_largest_connected_component() returns isolated LCC.
        :return:
        """
        data = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

        expected = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

        isolated = isolate_largest_connected_component(data)

        self.assertEqual(isolated.all(), expected.all())

    def test_crop_image_by_square_bbox_works(self):
        """
        When given an image and sq_bbox and size_width, it crops the image and resizes it.
        :return:
        """

        image = np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ])

        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        expected_image = np.array([
            [0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
        ])

        expected_image = expected_image.astype(np.uint8)
        expected_image = cv2.cvtColor(expected_image, cv2.COLOR_GRAY2BGR)

        rect_bbox = (1, 2, 3, 4)

        cropped = crop_image_by_square_bbox(image, rect_bbox, 6)

        self.assertEqual(cropped.all(), expected_image.all())

    def test_get_square_bbox(self):
        """
        When given an image and a rect bbox, get_square_bbox() returns the rectangular bbox.
        :return:
        """

        image = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])

        rect_bbox = (1, 1, 5, 6)
        sq_bbox = get_square_bbox(rect_bbox, image)
        self.assertEqual([0, 1, 5, 6], sq_bbox)


class TestDataMethods(unittest.TestCase):
    def test_split_data(self):
        # Tests that the split function shuffles data.
        inp1 = ([1, 2, 3, 4], 'a')
        inp2 = ([5, 6, 7, 8], 'b')
        inp3 = ([9, 10, 11, 12], 'c')
        x_train, x_test, y_train, y_test = split_data([inp1, inp2, inp3])
        self.assertNotEqual([
                            [1, 2, 3, 4, 5, 6, 7, 8, 9],
                            [10, 11, 12],
                            ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c'],
                            ['c', 'c', 'c']], y_train)


if __name__ == '__main__':
    unittest.main()
