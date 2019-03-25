import os

import numpy as np

from keras.models import load_model

from config import CONFIG

model = load_model(os.path.join(CONFIG['path_to_results'], CONFIG['predictor_model_dir'], 'model.hdf5'))


def normalize_list(l):
    l_sum = sum(l)
    norm_val = 100 / l_sum
    new_l = list()
    for entry in l:
        new_l.append(entry * norm_val)
    return list(new_l)


def predict(img):
    prediction = model.predict(np.array([img]))
    normalized = normalize_list(list(prediction[0]))

    idx = normalized.index(max(normalized))

    return idx + 1, normalized, CONFIG['classes'][idx]  # Class is from 1 to 5. No 0s.
