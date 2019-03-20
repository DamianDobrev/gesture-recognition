import cv2
import os
import numpy as np

from image_processing.image_processor import process_img_for_train_or_predict
# from loop import extract_bounding_boxes_by_skin_threshold

prefix_path = '_skip/'


def fetch_imgs_from_dir(data_dir, extension='png', num_entries=400):
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
        img = cv2.imread(data_dir + file_name)
        img = process_img_for_train_or_predict(img)
        images.append(img)

    # Uncomment below to visualise the first image and pause.
    # shit = np.array(images[0])
    # print('dasdsa', shit.shape)
    # shit = np.moveaxis(shit, -1, 0)
    # shit = np.moveaxis(shit, -1, 0)
    # print('dasdsa', shit.shape)
    # cv2.imshow('shittty', shit)
    # cv2.waitKey(0)
    return images


def fetch_training_images_binary(path=prefix_path, count=100):
    folders = os.listdir(path)
    classes = []
    for idx, name in enumerate(folders):
        class_num = idx
        classes.append((fetch_imgs_from_dir(path + name + '/', 'png', count), class_num))
    return classes
