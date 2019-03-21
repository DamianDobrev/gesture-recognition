import datetime
import time

import cv2

from keras import backend as K
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
# from sklearn.utils import shuffle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

import config
import data

import os


K.set_image_dim_ordering('th')

img_rows, img_cols = 50, 50

results_path = config.CONFIG['path_to_results']

batch_size = 16
num_epochs = 15

# Layers options.
num_conv_filters = 50
conv_kernel_size = 3
pool_size = 2


def visualizeHis(hist):
    # visualizing losses and accuracy

    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(num_epochs)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)

    plt.show()


def save_info(file_dir = '.'):
    # TODO (complete)
    f = open(file_dir + '/info.txt', 'w+')
    for i in range(10):
        f.write("This is line %d\r\n" % (i + 1))
    f.close()


def create_model(num_classes):
    global get_output
    model = Sequential()
    model.add(Conv2D(num_conv_filters, (conv_kernel_size, conv_kernel_size),
                     padding='valid',
                     input_shape=(1, img_rows, img_cols)))

    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(num_conv_filters, (conv_kernel_size, conv_kernel_size)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Save model.
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # Model summary
    model.summary()
    # Model conig details
    model.get_config()

    layer = model.layers[11]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])

    return model


def train_model(model, X_train, X_test, Y_train, Y_test):
    print('Started training...')

    print('~~~X_train~~~')
    print('ndim', X_train.ndim)
    print('shape', X_train.shape)
    print('size', X_train.size)
    print('len', len(X_train))

    print('~~~X_test~~~')
    print('ndim', X_test.ndim)
    print('shape', X_test.shape)
    print('size', X_test.size)
    print('len', len(X_test))

    print('start training...')
    hist = model.fit(X_train, np.array(Y_train), batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.2)
    print('finished training...')

    eval = model.evaluate(X_test, np.array(Y_test))

    print('eval:', eval)

    visualizeHis(hist)

    # Save the weights and model under _results/current-timestamp
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d__%H-%M-%S')
    os.makedirs(results_path + '/' + st)

    # Save weights
    model.save_weights(results_path + '/' + st + "/weight.hdf5", overwrite=True)
    # Save model
    model.save(results_path + '/' + st + "/model.hdf5")

    save_info(results_path + '/' + st)


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
