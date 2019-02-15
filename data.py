import cv2
import os

import numpy as np
from PIL import Image


def to_monochrome(img):
    monochrome_img = np.array(Image.fromarray(img).convert('L'))
    return monochrome_img


def read_and_process_img_from_file(file_path):
    img = cv2.imread(file_path)
    monochrome_img = to_monochrome(img)
    return monochrome_img


def fetch_imgs_from_dir(data_dir, extension='png'):
    """
    Returns np.array of monochrome images.
    :param data_dir: String showing the data dir.
    :param extension: String showing the file extension. 'png' by default.
    :return: np.array
    """

    def get_file_names_from_dir(data_dir, extension):
        all_png_imgs = []
        filelist = os.listdir(data_dir)
        for fichier in filelist[:]:  # filelist[:] makes a copy of filelist.
            if not (fichier.endswith('.' + extension)):
                filelist.remove(fichier)
        all_png_imgs.extend(filelist)

        return all_png_imgs

    images = []
    file_names = get_file_names_from_dir(data_dir, extension)

    for file_name in file_names:
        images.append(read_and_process_img_from_file(data_dir + file_name))

    images = np.array(images)

    return images


def fetch_data():
    general_path = './senz3d_dataset/acquisitions/'

    all_img = []
    for j in range(1, 4):
        data_dir = general_path + 'S' + str(j) + '/G' + str(j) + '/'
        all_img.extend(fetch_imgs_from_dir(data_dir, 'png'))

    return all_img


def fetch_data_test():
    general_path = './test_img/'
    return fetch_imgs_from_dir(general_path, 'png')


imgs = fetch_data_test()
print('len', len(imgs))
cv2.imshow('thing', imgs[0])

cv2.waitKey(0)