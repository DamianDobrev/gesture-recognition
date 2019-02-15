import cv2

from PIL import Image

from keras import backend as K
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
# from sklearn.utils import shuffle
from keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

import data

import os

from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
from tensorflow.python.keras.utils import np_utils

from image_processor import to_gray

K.set_image_dim_ordering('th')

img_rows, img_cols = 100, 100

img_channels = 1


# Batch_size to train
batch_size = 8 # 32

## Number of output classes (change it accordingly)
## eg: In my case I wanted to predict 4 types of gestures (Ok, Peace, Punch, Stop)
## NOTE: If you change this then dont forget to change Labels accordingly
nb_classes = 3

# Number of epochs to train (change it accordingly)
nb_epoch = 100  #25

# Total number of convolutional filters to use
nb_filters = 32
# Max pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3

classes = ['palm', 'peace', 'fist']

def create_model():
    global get_output
    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                     padding='valid',
                     input_shape=(1, img_rows, img_cols)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
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


def visualizeHis(hist):
    # visualizing losses and accuracy

    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(nb_epoch)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    #plt.style.use(['classic'])

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)

    plt.show()


def train_model(model, X_train, X_test, Y_train, Y_test):
    print('Started training...')

    X_train = X_train.reshape(len(X_train), 1, img_rows, img_cols)
    X_test = X_test.reshape(len(X_test), 1, img_rows, img_cols)
    # X_test_new = X_train.reshape(len(X_test), img_rows, img_cols, 1)
    #
    # # X_train_new = np.array(X_train_new)
    # # X_test_new = np.array(X_test_new)
    #
    print('~~~X_train~~~')
    print('ndim', X_train.ndim)
    print('shape', X_train.shape)
    print('size', X_train.size)
    print('len', len(X_train))
    #
    print('~~~X_test~~~')
    print('ndim', X_test.ndim)
    print('shape', X_test.shape)
    print('size', X_test.size)
    print('len', len(X_test))


    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                     verbose=1, validation_split=0.2)
    print('finished training.')

    eval = model.evaluate(X_test, Y_test)

    print('eval:', eval)

    visualizeHis(hist)

    model.save_weights("newWeight.hdf5", overwrite=True)

    # Save model as well
    model.save("newModel.hdf5")
# %%


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

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    x_train, x_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=4)
    print('y_train',y_train)
    print('np.array(y_train',np.array(y_train))
    print('y_test',y_test)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


# Data fetching
# ...
imgs_palm = data.fetch_data(1)
imgs_peace = data.fetch_data(2)
imgs_fist = data.fetch_data(5)
# print('len', len(imgs))
# for j, img in enumerate(imgs[25:35]):
#     cv2.imshow('thing' + str(j), img)


# Splitting data to labels
# ...
# X_train, X_test, Y_train, Y_test = split_data(imgs)
X_train, X_test, Y_train, Y_test = split_data([(imgs_palm, 0), (imgs_peace, 1), (imgs_fist, 2)])
# print('\n\n\n\n\ntrain\n\n\n\n')
# print(len(X_train))
# print(Y_train)
# print('\n\n\n\n\ntest\n\n\n\n')
# print(len(X_test))
# print(Y_test)


# Train model
# ...
# TODO
model = create_model()
ones = np.full(2, 7)
print(ones)
train_model(model, X_train, X_test, Y_train, Y_test)


cv2.waitKey(0)