import cv2
import os
import numpy as np

from image_processing.image_processor import to_50x50_monochrome


def visualise_img_and_pause(img):
    img_np = np.array(img)
    # print('Image shape as it is:', img_np.shape)
    # img_np = np.moveaxis(img_np, -1, 0)
    # img_np = np.moveaxis(img_np, -1, 0)
    # print('Image shape to visualise:', img_np.shape)
    cv2.imshow('Image', img_np)
    cv2.waitKey(0)


def fetch_imgs_from_dir(data_dir, extension='png', max_count=100):
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
        if idx >= max_count:
            break
        img = cv2.imread(os.path.join(data_dir,file_name))
        # img = to_50x50_monochrome(img)
        images.append(img)

    # Uncomment this to test how the image looks like.
    # visualise_img_and_pause(images[0])
    return images


def fetch_training_images(path, max_imgs_per_class):
    folders = os.listdir(path)
    classes = []
    for idx, name in enumerate(folders):
        class_num = idx
        images = fetch_imgs_from_dir(os.path.join(path, name), 'png', max_imgs_per_class)
        classes.append((images, class_num))
    return classes
