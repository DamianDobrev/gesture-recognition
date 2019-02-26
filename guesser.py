import cv2

import numpy as np
import model as md
from keras import backend as K

import image_processor as ip

img_rows, img_cols = 200, 200

img_channels = 1

def guessGesture(model, img):
    global output, get_output, jsonarray
    # Load image and flatten it
    image = np.array(img).flatten()

    # reshape it
    # cv2.imshow('before reshape', image)
    # cv2.waitKey(0)
    #
    # image = image.reshape(img_channels, img_rows, img_cols)
    #
    # # float32
    # image = image.astype('float32')
    #
    # # normalize it
    # image = image / 255
    #
    # # reshape for NN
    # rimage = image.reshape(1, img_channels, img_rows, img_cols)

    # Now feed it to the NN, to fetch the predictions
    # index = model.predict_classes(rimage)
    # prob_array = model.predict_proba(rimage)

    prob_array = get_output([img, 0])[0]

    # print prob_array

    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1

    # Get the output with maximum probability
    import operator

    guess = max(d.items(), key=operator.itemgetter(1))[0]
    prob = d[guess]

    if prob > 60.0:
        # print(guess + "  Probability: ", prob)

        # Enable this to save the predictions in a json file,
        # Which can be read by plotter app to plot bar graph
        # dump to the JSON contents to the file

        # with open('gesturejson.txt', 'w') as outfile:
        #    json.dump(d, outfile)
        jsonarray = d

        return output.index(guess)

    else:
        return 1


img = cv2.imread('training_img_crop/skin001.png')
# cv2.imshow('111111',img)
# img = ip.to_gray(img)
# cv2.imshow('222222',img)
# print('start guessing')
#
# model = md.create_model()
# layer = model.layers[11]
# get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
# guessed = guessGesture(model, img)
# print('guessed')
# print(guessed)

cv2.waitKey(0)
