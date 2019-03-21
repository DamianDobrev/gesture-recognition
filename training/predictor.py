import os

import numpy as np

from keras.models import load_model

from config import CONFIG
from image_processing.image_processor import to_50x50_monochrome

model_folder_name = '2019-03-18__23-55-36'  # 100 images per class, 5 classes

model = load_model(os.path.join(CONFIG['path_to_results'], model_folder_name, 'model.hdf5'))

# !!! Note.
# This model works very well: '2019-03-18__23-55-36'.
# However it only has 5 classes, and they are in different order.
# If one would want to test it, the classes from CONFIG['classes'] have to be replaced with this array:
# ['stop', 'fist', 'right', 'left', 'updown']


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
