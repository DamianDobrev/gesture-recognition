import os


def create_path_if_does_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
