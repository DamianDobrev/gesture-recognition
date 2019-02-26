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


def fetch_data(gesture_label = 1):
    general_path = './_skip/senz3d_dataset/acquisitions/'

    all_img = []
    for j in range(1, 4):
        data_dir = general_path + 'S' + str(j) + '/G' + str(gesture_label) + '/'
        all_img.extend(fetch_imgs_from_dir(data_dir, 'png'))

    all_img = [crop_img(x) for x in all_img]
    return all_img


def fetch_data_test():
    general_path = './_skip/test_img/'
    return fetch_imgs_from_dir(general_path, 'png')


def crop_img(img, w=100, h=100):
    im = Image.fromarray(img)

    # Resize the image so that the bounding box of (w, h) is completely filled!
    width = im.width
    height = im.height
    ratio_w = width / w
    ratio_h = height / h
    min_ratio = min(ratio_w, ratio_h)
    new_width = int(width / min_ratio)
    new_height = int(height / min_ratio)
    im = im.resize((new_width, new_height))

    # Get the ideal center.
    left = (new_width - w) / 2
    top = (new_height - h) / 2
    right = (new_width + w) / 2
    bottom = (new_height + h) / 2
    im = im.crop((left, top, right, bottom))

    return np.array(im)



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