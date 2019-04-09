import datetime
import time

from keras import backend as K
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from keras.callbacks import CSVLogger, EarlyStopping

import config

import os

# Just to specify that the images have to be provided in the model in format (X, Y, channels).
from modules.common import create_path_if_does_not_exist
from modules.cnn.save_data import save_notes, save_data_info, save_hist, save_info, save_config, save_confusion_matrix

K.set_image_dim_ordering('tf')

img_rows, img_cols = config.CONFIG['training_img_size'], config.CONFIG['training_img_size']
batch_size = config.CONFIG['batch_size']
num_epochs = config.CONFIG['num_epochs']
results_path = config.CONFIG['results_path']


def create_model(num_classes):
    """
    Creates a CNN model.
    :param num_classes: Number of classes this model will accept.
    :return: Does not return anything.
    """

    # Layers options.
    num_conv_filters = 60
    conv_kernel = (3, 3)
    max_pooling_pool_size = (2, 2)

    model = Sequential()

    layers = [
        Conv2D(num_conv_filters, conv_kernel,
               padding='valid',
               activation='relu',
               input_shape=(img_rows, img_cols, 1)),
        Conv2D(num_conv_filters, conv_kernel, activation='relu'),
        MaxPooling2D(pool_size=max_pooling_pool_size),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ]

    for layer in layers:
        model.add(layer)

    # Save model.
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # Print model summary and config details.
    model.summary()
    model.get_config()

    return model


def train_model(model, x_train, x_test, y_train, y_test):
    """
    Trains a CNN model.
    Saves all relevant information and configuration in the same folder
    as the trained model.
    :param model: A model to train.
    :param x_train: Training set.
    :param x_test: Test set.
    :param y_train: Training labels.
    :param y_test: Test labels.
    :return:
    """
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Save the weights and model under _results/current-timestamp
    create_path_if_does_not_exist(results_path)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d__%H-%M-%S')
    create_path_if_does_not_exist(os.path.join(results_path, st))
    save_notes(os.path.join(results_path, st))
    print('saving data...')
    save_data_info(os.path.join(results_path, st), x_train, x_test, y_train, y_test)

    print('started training...')
    csv_logger = CSVLogger(os.path.join(results_path, st, 'model_fit_log.csv'), append=True, separator=';')
    # Add early stopping because the model may reach super-high accuracy in ~5 epochs
    # but if we continue with training it will overfit super hard.
    es = EarlyStopping(monitor='acc', mode='max', verbose=1, patience=5)
    hist = model.fit(x_train, y_train, batch_size=batch_size,
                     epochs=num_epochs, verbose=1, validation_split=config.CONFIG['validation_split'],
                     callbacks=[csv_logger, es])
    print('finished training...')

    eval = model.evaluate(x_test, y_test)
    print('eval:', eval)

    # Save weights and model.
    print('saving weights...')
    model.save_weights(os.path.join(results_path, st, "weight.hdf5"), overwrite=True)
    print('saving model...')
    model.save(os.path.join(results_path, st, "model.hdf5"), overwrite=True)

    # Save model and training info.
    path = os.path.join(results_path, st)
    print('saving config...')
    save_config(path)
    print('saving info...')
    save_info(path, model, eval)
    print('saving histograms...')
    save_hist(hist, path)
    print('saving confusion matrix...')
    prediction = model.predict(x_test)
    save_confusion_matrix(path, prediction, y_test)


def split_data(data_label_tuples):
    """
    Splits data into training and test sets. Also splits the
    labels.
    :param data_label_tuples: Data tuples in format:
        [ ([img1.png, img2.png], 0)
          ([img3.png, img4.png], 1) ]
    :return:
        - x_train list of images
        - x_test list of images
        - y_train list of labels
        - y_test list of labels
    """
    all_data = []
    all_labels = []

    for tup in data_label_tuples:
        data = tup[0]
        label = tup[1]
        print('Class [' + str(label) + '] contains ' + str(len(data)) + ' samples.')
        labels = np.full(len(data), label)
        data, labels = shuffle(data, labels)
        all_data.extend(data)
        all_labels.extend(labels)

    all_data, all_labels = shuffle(all_data, all_labels, random_state=2)
    x_train, x_test, y_train, y_test = \
        train_test_split(all_data,
                         all_labels,
                         test_size=config.CONFIG['test_split'],
                         random_state=4)

    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def to_np_array(list_to_transform):
    shape = list(list_to_transform[0].shape)
    shape[:0] = [len(list_to_transform)]
    return np.concatenate(list_to_transform).reshape(shape)

#
# inp1 = ([1,2,3,4], 'a')
# inp2 = ([5,6,7,8], 'b')
# inp3 = ([9,10,11,12], 'c')
# split_data([inp1,inp2,inp3])
