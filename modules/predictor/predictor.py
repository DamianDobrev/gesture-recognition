import os

import numpy as np

from keras.models import load_model

from config import CONFIG

model = load_model(os.path.join(CONFIG['results_path'], CONFIG['predictor_model_dir'], 'model.hdf5'))


def normalize_list(l):
    """
    Normalizes a list of probabilities to add up to 100.
    :param l: A list of probabilities that do not add up to 100.
    :return: Normalized list.
    """
    l_sum = sum(l)
    norm_val = 100 / l_sum
    new_l = list()
    for entry in l:
        new_l.append(entry * norm_val)
    return list(new_l)


def predict(img):
    """
    Predicts a gesture in an image.
    :param img: A BGR image with shape (X,Y,3).
    :return:
        - Predicted class number (class idx + 1).
        - Normalized predictions.
        - Predicted label.
    """
    prediction = model.predict(np.array([img]))
    normalized = normalize_list(list(prediction[0]))

    idx_of_max = normalized.index(max(normalized))

    # If the best probability is less than our threshold, nothing
    # is predicted.
    if normalized[idx_of_max] < CONFIG['predicted_val_threshold']:
        return -1, normalized, '--none--'

    return idx_of_max + 1, normalized, CONFIG['classes'][idx_of_max]
