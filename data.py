import cv2
import os

from config import CONFIG


def fetch_saved_hsv():
    f = open(CONFIG['path_to_hsv_ranges_csv'], 'r')
    f.readline()
    l_vals = f.readline().split(',')
    u_vals = f.readline().split(',')
    f.close()
    l_range = [int(l_vals[0]), int(l_vals[1]), int(l_vals[2])]
    u_range = [int(u_vals[0]), int(u_vals[1]), int(u_vals[2])]
    return l_range, u_range


def fetch_imgs_from_dir(data_dir, extension='png', max_count=100):
    """
    Fetches files from a directory with the specified extension.
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
        images.append(img)

    return images


def fetch_training_images(path, max_imgs_per_class):
    folders = os.listdir(path)
    classes = []
    for idx, name in enumerate(folders):
        images = fetch_imgs_from_dir(os.path.join(path, name), 'png', max_imgs_per_class)
        classes.append((images, int(name) - 1))
    return classes
