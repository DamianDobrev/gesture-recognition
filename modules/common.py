import os


def create_path_if_does_not_exist(path):
    """
    Helper method to create path if it does not exist.
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
