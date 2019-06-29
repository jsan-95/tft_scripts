import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import datetime
import pickle

from keras.callbacks import EarlyStopping
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import random


DATA_TRAIN_WHITE_BLACK = "/Volumes/Seagate Expansion Drive/TFT/complete_dataset"
DATA_TEST = "data/test/test_folder"
CATEGORIES = []
NUM_CLASSES = 10
IMG_SIZE = 50
EPOCHS = 25
PICKLE = False

training_data = []
validation_data = []
test_data = []


def create_training_data(training_data=None, validation_data=None, test_data=None):
    i = 0
    # data = []
    for category in CATEGORIES:
        data = []
        path = os.path.join(DATA_TRAIN_WHITE_BLACK, category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([new_array, class_num])
            except Exception as e:
                pass

        img_total = os.listdir(path).__len__()
        first = int(img_total * 0.8 * 0.8)
        second = int(np.ceil(img_total * 0.8))

        random.shuffle(data)
        training_data += data[:first]
        validation_data += data[first:second]
        test_data += data[second:]
        i += 1
        if i % 50 == 0:
            print(str(i)+" imagenes")

def model_first():
    model = Sequential()
    model.add(Conv2D(256, (5, 5), input_shape=X_TRAIN.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    return model


def model_V2():
    model = Sequential()
    model.add(Conv2D(256, (5, 5), input_shape=X_TRAIN.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    return model


def model_V3():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), input_shape=X_TRAIN.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8, (2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(4, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    return model


def model_V4():
    model = Sequential()
    model.add(Conv2D(256, (5, 5), input_shape=X_TRAIN.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    return model


def model_V5():
    model = Sequential()
    model.add(Conv2D(16, (5, 5), input_shape=X_TRAIN.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    return model


def model_V6():
    model = Sequential()
    model.add(Conv2D(16, (2, 2), input_shape=X_TRAIN.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (2, 2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (2, 2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    return model

def last_model_1000():
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

    return model


def plotACC(history):
    fig, ax = plt.subplots()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    ax.xaxis.set_ticks(np.arange(0, len(history.history['loss']) + 1, 1))
    plt.grid(True)
    plt.show()


def plotLOSS(history):
    fig, ax = plt.subplots()
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    ax.xaxis.set_ticks(np.arange(0, len(history.history['loss']) + 1, 1))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)
    plt.show()


def createCategories():
    i = 0
    for CLASS in os.listdir(DATA_TRAIN_WHITE_BLACK):
        if os.path.isdir(os.path.join(DATA_TRAIN_WHITE_BLACK, CLASS)):
            CATEGORIES.append(CLASS)
            i += 1
        if i == NUM_CLASSES:
            break
    print('---------ID MUESTRAS---------', CATEGORIES)


def createPickles(X_TRAIN,Y_TRAIN,X_VALIDATION,Y_VALIDATION,X_TEST,Y_TEST):
    pickle_out = open("100_X_TRAIN.pickle", "wb")
    pickle.dump(X_TRAIN, pickle_out)
    pickle_out.close()
    pickle_out = open("100_Y_TRAIN.pickle", "wb")
    pickle.dump(Y_TRAIN, pickle_out)
    pickle_out.close()
    pickle_out = open("100_X_VALIDATION.pickle", "wb")
    pickle.dump(X_VALIDATION, pickle_out)
    pickle_out.close()
    pickle_out = open("100_Y_VALIDATION.pickle", "wb")
    pickle.dump(Y_VALIDATION, pickle_out)
    pickle_out.close()
    pickle_out = open("100_X_TEST.pickle", "wb")
    pickle.dump(X_TEST, pickle_out)
    pickle_out.close()
    pickle_out = open("100_Y_TEST.pickle", "wb")
    pickle.dump(Y_TEST, pickle_out)
    pickle_out.close()

def createPicklesManrique(X_TRAIN,Y_TRAIN,X_VALIDATION,Y_VALIDATION,X_TEST,Y_TEST):
    pickle_out = open("MANRIQUE_X_TRAIN.pickle", "wb")
    pickle.dump(X_TRAIN, pickle_out)
    pickle_out.close()
    pickle_out = open("MANRIQUE_Y_TRAIN.pickle", "wb")
    pickle.dump(Y_TRAIN, pickle_out)
    pickle_out.close()
    pickle_out = open("MANRIQUE_X_VALIDATION.pickle", "wb")
    pickle.dump(X_VALIDATION, pickle_out)
    pickle_out.close()
    pickle_out = open("MANRIQUE_Y_VALIDATION.pickle", "wb")
    pickle.dump(Y_VALIDATION, pickle_out)
    pickle_out.close()
    pickle_out = open("MANRIQUE_X_TEST.pickle", "wb")
    pickle.dump(X_TEST, pickle_out)
    pickle_out.close()
    pickle_out = open("MANRIQUE_Y_TEST.pickle", "wb")
    pickle.dump(Y_TEST, pickle_out)
    pickle_out.close()


if __name__ == '__main__':

    print("-----------COMIENZO----------, ", datetime.datetime.now())

    X_TRAIN = []
    Y_TRAIN = []
    X_VALIDATION = []
    Y_VALIDATION = []
    X_TEST = []
    Y_TEST = []

    createCategories()
    print(CATEGORIES)

    if not PICKLE:
        createCategories()

        create_training_data(training_data, validation_data, test_data)
        for features, label in training_data:
            X_TRAIN.append(features)
            Y_TRAIN.append(label)
        for features, label in validation_data:
            X_VALIDATION.append(features)
            Y_VALIDATION.append(label)
        for features, label in test_data:
            X_TEST.append(features)
            Y_TEST.append(label)
        X_TRAIN = np.array(X_TRAIN).reshape(np.shape(X_TRAIN)[0], IMG_SIZE, IMG_SIZE, 1)
        X_VALIDATION = np.array(X_VALIDATION).reshape(np.shape(X_VALIDATION)[0], IMG_SIZE, IMG_SIZE, 1)
        X_TEST = np.array(X_TEST).reshape(np.shape(X_TEST)[0], IMG_SIZE, IMG_SIZE, 1)
        createPickles(X_TRAIN,Y_TRAIN,X_VALIDATION,Y_VALIDATION,X_TEST,Y_TEST)
    else:
        pickle_in = open("100_X_TRAIN.pickle", "rb")
        X_TRAIN = pickle.load(pickle_in)
        pickle_in = open("100_Y_TRAIN.pickle", "rb")
        Y_TRAIN = pickle.load(pickle_in)
        pickle_in = open("100_X_VALIDATION.pickle", "rb")
        X_VALIDATION = pickle.load(pickle_in)
        pickle_in = open("100_Y_VALIDATION.pickle", "rb")
        Y_VALIDATION = pickle.load(pickle_in)
        pickle_in = open("100_X_TEST.pickle", "rb")
        X_TEST = pickle.load(pickle_in)
        pickle_in = open("100_Y_TEST.pickle", "rb")
        Y_TEST = pickle.load(pickle_in)

    total_data = len(X_TRAIN) + len(X_VALIDATION) + len(X_TEST)


    print('-------Size of data-----')
    print('TRAIN: ', len(X_TRAIN), ' PORCENTAJE: ', len(X_TRAIN) / total_data * 100)
    print('VALIDATION: ', len(X_VALIDATION), ' PORCENTAJE: ', len(X_VALIDATION) / total_data * 100)
    print('TEST: ', len(X_TEST), ' PORCENTAJE: ', len(X_TEST) / total_data * 100)
    print('---------X SHAPE---------', np.shape(X_TRAIN))

    sys.exit()
    # Transformar a vector onehot
    a = np.array(Y_VALIDATION)
    Y_VALIDATION = np.zeros((len(a), NUM_CLASSES))
    Y_VALIDATION[np.arange(len(a)), a] = 1

    a = np.array(Y_TRAIN)
    Y_TRAIN = np.zeros((len(a), NUM_CLASSES))
    Y_TRAIN[np.arange(len(a)), a] = 1

    a = np.array(Y_TEST)
    Y_TEST = np.zeros((len(a), NUM_CLASSES))
    Y_TEST[np.arange(len(a)), a] = 1

    X_TRAIN = X_TRAIN / 255.0
    X_VALIDATION = X_VALIDATION / 255.0
    X_TEST = X_TEST / 255.0

    model = model_first()
    # model = last_model_1000()

    print('---------X SHAPE---------', np.shape(X_TRAIN))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=2)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])
    history = model.fit(X_TRAIN, Y_TRAIN,
                        batch_size=32,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=(X_VALIDATION, Y_VALIDATION),
                        callbacks=[es])
    score = model.evaluate(X_TEST, Y_TEST, batch_size=32)
    # model.save_weights('my_model_last_weights_manrique.h5')
    print(score)
    print(history)

    plotACC(history)
    plotLOSS(history)

    print("-----FINAL-----", datetime.datetime.now())
