import numpy as np
import os
import sys
import cv2
import tensorflow
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D



DATA_TEST = "/Volumes/Seagate Expansion Drive/TFT/Conjunto de datos/resized_squares"

NUM_CLASSES = 8
IMG_SIZE = 50
CATEGORIES = ['1', '17', '18', '2', '3', '4', '8', '9']

test_data = []


def create_test_data():
    for i in range(0, NUM_CLASSES):
        try:

            img = os.path.join(DATA_TEST, CATEGORIES[i] + '.jpg')
            class_num = CATEGORIES.index(CATEGORIES[i])
            img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            print(img)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            test_data.append([new_array, class_num])

        except Exception as e:
            pass
    print(np.shape(test_data))
    X_TEST = []
    Y_TEST = []
    for features, label in test_data:
        X_TEST.append(features)
        Y_TEST.append(label)
    # pickle_out = open("100_X_TEST_RESIZE_SQUARES.pickle", "wb")
    # pickle.dump(X_TEST, pickle_out)
    # pickle_out.close()
    # pickle_out = open("100_Y_TEST_RESIZE_SQUARES.pickle", "wb")
    # pickle.dump(Y_TEST, pickle_out)

def create_model():

    # -------V6-------
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(50, 50, 1)))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))

    model.add(Conv2D(16, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    #  V2
    # model.load_weights("my_model_weights_pillow.h5")
    # model.load_weights("models/last_model_1000_weights22.h5")
    model.load_weights("my_model_last_weights_manrique.h5")

    return model


if __name__ == '__main__':

    errores = 0
    res = ""
    create_test_data()

    for i in range(0, NUM_CLASSES):
        print("iter: " + str(i))
        X_TEST = []
        Y_TEST = []

        X_TEST.append(test_data[i][0])
        Y_TEST.append(test_data[i][1])

        X_TEST = np.array(X_TEST).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        a = np.array(Y_TEST)
        Y_TEST = np.zeros((len(a), NUM_CLASSES))
        Y_TEST[np.arange(len(a)), a] = 1

        X_TEST = X_TEST / 255.0

        model = create_model()

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])
        score = model.predict(X_TEST, batch_size=1, verbose=0)
        index = np.argmax(score, axis=1)[0]
        id = CATEGORIES[index]

        res += CATEGORIES[i] + ":" + id + "\n"
        # if CATEGORIES[i] != id:
        #     res += CATEGORIES[i] + ":" + id + "\n"
        #     errores += 1
        #     print(res)


    print(errores)
    print(res)
    sys.exit()
