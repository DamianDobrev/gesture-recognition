import numpy as np

from keras.models import load_model

from data import process_img_for_train_or_predict

model_folder_name = '2019-03-18__23-55-36'
model = load_model('../__results/' + model_folder_name + '/model.hdf5')

classes_to_text = {
    0: 'STOP',
    1: 'FIST',
    2: 'RIGHT',
    3: 'LEFT',
    4: 'UPDOWN'
}


def normalize_list(l):
    l_sum = sum(l)
    print('sum:', l_sum)
    norm_val = 100 / l_sum
    new_l = list()
    for entry in l:
        new_l.append(entry * norm_val)
    return list(new_l)


def predict(img):
    img = process_img_for_train_or_predict(img)

    prediction = model.predict(np.array([img]))
    normalized = normalize_list(list(prediction[0]))
    print('normalized:', normalized)

    idx = normalized.index(max(normalized))
    print(classes_to_text[idx])

    return idx + 1, normalized, classes_to_text[idx]  # Class is from 1 to 5. No 0s.
