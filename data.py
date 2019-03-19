import cv2
import os

import numpy as np
from PIL import Image

prefix_path = '_skip/'

def to_monochrome(img):
    monochrome_img = np.array(Image.fromarray(img).convert('L'))
    return monochrome_img


def read_and_process_img_from_file(file_path):
    img = cv2.imread(file_path)
    img = cv2.resize(img, (50, 50))
    print(file_path, img.shape)
    img = to_monochrome(img)
    img = np.array([img])
    print('imgshape', img.shape)  # should be like (1, 50, 50)
    return img


def fetch_imgs_from_dir(data_dir, extension='png', num_entries=100):
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

    for idx, file_name in enumerate(file_names):
        if idx >= num_entries:
            break
        img = read_and_process_img_from_file(data_dir + file_name)
        images.append(img)

    return images


def fetch_training_images_binary(path=prefix_path):
    files = os.listdir(path)
    classes = []
    for name in files:
        class_num = int(name) - 1
        classes.append((fetch_imgs_from_dir(path + name + '/', 'png'), class_num))
    return classes



# img1 = fetch_data(1)[0]
# img2 = fetch_data(2)[36]
# img3 = fetch_data(3)[69]
# img4 = fetch_data(4)[12]
# img5 = fetch_data(5)[77]
# # print('len', len(imgs))
# # cv2.imshow('thing', imgs[0])
#
# cv2.imshow('cropped1', crop_img(img1))
# cv2.imshow('cropped2', crop_img(img2))
# cv2.imshow('cropped3', crop_img(img3))
# cv2.imshow('cropped4', crop_img(img4))
# cv2.imshow('cropped5', crop_img(img5))
#
# cv2.waitKey(0)