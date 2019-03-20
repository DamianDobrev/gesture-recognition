import cv2
import os

from image_processing.image_processor import process_img_for_train_or_predict

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

    return images


def fetch_training_images_binary(path=prefix_path):
    files = os.listdir(path)
    classes = []
    for name in files:
        class_num = int(name) - 1
        classes.append((fetch_imgs_from_dir(path + name + '/', 'png'), class_num))
    return classes