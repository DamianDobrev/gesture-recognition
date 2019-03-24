import datetime
import time
import random

import cv2

from keras import backend as K
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from keras.callbacks import CSVLogger

import config

import os


K.set_image_dim_ordering('th')

img_rows, img_cols = 50, 50

results_path = config.CONFIG['path_to_results']

batch_size = 32
num_epochs = 20


def create_path_if_does_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_hist(hist, path_to_save):
    # visualizing losses and accuracy

    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(num_epochs)

    fig_loss = plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])

    fig_acc = plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)

    plt.show()
    fig_acc.savefig(os.path.join(path_to_save, 'accuracy.png'))
    fig_loss.savefig(os.path.join(path_to_save, 'loss.png'))


def save_data_info(file_dir, x_train, x_test, y_train, y_test):
    data_path = os.path.join(file_dir, 'data')
    create_path_if_does_not_exist(data_path)

    f = open(os.path.join(file_dir, 'data', 'data_info.txt'), 'w+')
    f.write('len(x_train): ' + str(len(x_train)) + '\n')
    f.write('len(x_test): ' + str(len(x_test)) + '\n')
    f.close()

    for idx, img in enumerate(x_train):
        img_path = os.path.join(data_path, str(config.CONFIG['classes'][y_train[idx]]))
        create_path_if_does_not_exist(img_path)
        conv = np.moveaxis(img, 0, -1)
        cv2.imwrite(os.path.join(img_path, 'img-' + str(random.randrange(999999)) + '.png'), conv)


def save_additional_info_file(file_dir):
    f = open(os.path.join(file_dir, 'notes.txt'), 'w+')
    f.write('# Add notes here about this training/model...' + '\n')
    f.close()


def save_info(file_dir, model, eval):
    f = open(os.path.join(file_dir, 'eval.txt'), 'w+')
    f.write('eval_loss=' + str(eval[0]) + '\n')
    f.write('eval_acc =' + str(eval[1]) + '\n\n')
    f.close()

    f = open(os.path.join(file_dir, 'model_summary.txt'), 'w+')
    f.write(str(model.summary(print_fn=lambda x: f.write(x + '\n'))))
    f.close()

    f = open(os.path.join(file_dir, 'training_summary.txt'), 'w+')
    f.write('img_w,img_h=' + str(img_cols) + ',' + str(img_rows) + '\n')
    f.write('batch_size=' + str(batch_size) + '\n')
    f.write('num_epochs=' + str(num_epochs) + '\n')
    f.write('classes=' + str(config.CONFIG['classes']))
    f.close()


def create_model(num_classes):
    # Layers options.
    num_conv_filters = 50
    conv_kernel_size = 3
    pool_size = 2

    global get_output
    model = Sequential()
    l_input = Conv2D(num_conv_filters, (conv_kernel_size, conv_kernel_size),
                     padding='valid',
                     input_shape=(1, img_rows, img_cols))
    model.add(l_input)

    layers = [
        Activation('relu'),
        Conv2D(num_conv_filters, (conv_kernel_size, conv_kernel_size)),
        Activation('relu'),
        MaxPooling2D(pool_size=(pool_size, pool_size)),
        Dropout(0.5),

        Flatten(),
        Dense(128),
        Activation('relu'),
        Dropout(0.5),
        Dense(num_classes),
        Activation('softmax')
    ]

    for layer in layers:
        model.add(layer)

    # Save model.
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # Model summary
    model.summary()
    # Model config details
    model.get_config()

    layer = model.layers[11]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])

    return model


def train_model(model, X_train, X_test, Y_train, Y_test):
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    # Save the weights and model under _results/current-timestamp
    create_path_if_does_not_exist(results_path)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d__%H-%M-%S')
    create_path_if_does_not_exist(os.path.join(results_path, st))
    save_additional_info_file(os.path.join(results_path, st))

    print('started training...')
    csv_logger = CSVLogger(os.path.join(results_path, st, 'model_fit_log.csv'), append=True, separator=';')
    hist = model.fit(X_train, Y_train, batch_size=batch_size,
                     epochs=num_epochs, verbose=1, validation_split=0.2,
                     callbacks=[csv_logger])
    print('finished training...')

    eval = model.evaluate(X_test, Y_test)
    print('eval:', eval)

    # Save weights and model.
    model.save_weights(os.path.join(results_path, st, "weight.hdf5"), overwrite=True)
    model.save(os.path.join(results_path, st, "model.hdf5"), overwrite=True)

    # Save model and training info.
    save_hist(hist, os.path.join(results_path, st))
    save_info(os.path.join(results_path, st), model, eval)
    save_data_info(os.path.join(results_path, st), X_train, X_test, Y_train, Y_test)


def split_data(data_label_tuples):
    all_data = []
    all_labels = []

    for tuple in data_label_tuples:
        data = tuple[0]
        label = tuple[1]
        labels = np.full(len(data), label)
        data, Label = shuffle(data, labels, random_state=2)
        all_data.extend(data)
        all_labels.extend(Label)

    x_train, x_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=4)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def to_np_array(list_to_transform):
    shape = list(list_to_transform[0].shape)
    shape[:0] = [len(list_to_transform)]
    return np.concatenate(list_to_transform).reshape(shape)
    # return np.concatenate(list_to_transform, axis=0)
