import cv2

from PIL import Image

from keras import backend as K
from keras import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
# from sklearn.utils import shuffle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

import os

from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
from tensorflow.python.keras.utils import np_utils

from image_processor import to_gray

# Local modules.
import data

K.set_image_dim_ordering('th')

img_rows, img_cols = 200, 200

img_channels = 1


# Batch_size to train
batch_size = 8 # 32

## Number of output classes (change it accordingly)
## eg: In my case I wanted to predict 4 types of gestures (Ok, Peace, Punch, Stop)
## NOTE: If you change this then dont forget to change Labels accordingly
nb_classes = 2

# Number of epochs to train (change it accordingly)
nb_epoch = 15  #25

# Total number of convolutional filters to use
nb_filters = 32
# Max pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3

path = "./"
path2 = './training_img_crop'

classes = ['peace', 'nothing']

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

    # Now start the training of the loaded model
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                     verbose=1, validation_split=0.2)

    visualizeHis(hist)

    model.save_weights("newWeight.hdf5", overwrite=True)

    # Save model as well
    model.save("newModel.hdf5")
# %%


def split_data(data):
    label = classes[1] # Zeros...
    data, Label = shuffle(data, np.zeros(len(data)), random_state=2)
    train_data = [data, Label]


    (X, y) = (train_data[0], train_data[1])

    # Split X and y into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalize
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    # These on top should be used
    return X_train, X_test, y_train, y_test


# %%
def modlistdir(path):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        # This check is to ignore any hidden files/folders
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist


def fetch_data():
    imlist = modlistdir(path2)

    image1 = np.array(Image.open(path2 + '/' + imlist[0]))  # open one image to get size
    # cv2.imshow('asd', image1)

    m, n = image1.shape[0:2]  # get the size of the images
    total_images = len(imlist)  # get the 'total' number of images

    # create matrix to store all flattened images
    immatrix = np.array([np.array(Image.open(path2 + '/' + images).convert('L')).flatten()
                         for images in sorted(imlist)], dtype='f')

    return immatrix


def train_img_crop():
    imgs = data.fetch_data()
    X_train, X_test, Y_train, Y_test = split_data(imgs)
    train_model(create_model(), X_train, X_test, Y_train, Y_test)
    # print(immatrix.shape)


# fetch_data()
train_img_crop()

cv2.waitKey(0)