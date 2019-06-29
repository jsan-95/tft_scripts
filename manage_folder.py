import sys
import fnmatch

from directoryPath import Directory
import cv2 as cv
import os
from matplotlib import pyplot as plt
import datetime
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from process import averaging
from process import blur
from process import brightness
from process import noise
from process import rotate
from process import translate


def displayPlots(img, dst, title):
    plt.subplot(121), plt.imshow(img), plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dst), plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show()


def writeImage(img, fn, name_class, dst, number):
    file_name = fn[:-4]
    main_folder = Directory.IMAGES_FILTERED + file_name

    if not os.path.isdir(main_folder):
        os.mkdir(main_folder)

    folder_class = os.path.join(main_folder, name_class)
    if not os.path.isdir(folder_class):
        os.mkdir(folder_class)

    folder_dst = folder_class + file_name + number + '.jpg'

    if folder_class:
        # displayPlots(img, dst, 'rotate')
        cv.imwrite(folder_dst, dst)


def applyFilters(img, fn):
    angles_rotate = [45, 60, 75, 105, 120, 135]
    for i in range(0, 6):
        writeImage(img, fn, 'averaging/', averaging.averaging(img), '-' + str(i * 1 + 1))
        writeImage(img, fn, 'blur/', blur.blur(img), '-' + str(i * 1 + 1))
        writeImage(img, fn, 'brightness/', brightness.brightness(img), '-' + str(i * 1 + 1))
        writeImage(img, fn, 'noise/', noise.noise(img), '-' + str(i * 1 + 1))
        writeImage(img, fn, 'rotate/', rotate.rotate(img, angles_rotate[i], 90, 90), '-' + str(i * 2 + 1))
        writeImage(img, fn, 'rotate/', rotate.rotate(img, 90, angles_rotate[i], 90), '-' + str(i * 2 + 2))
        writeImage(img, fn, 'rotate/', rotate.rotate(img, 90, angles_rotate[i], 90), '-' + str(i * 2 + 3))
        writeImage(img, fn, 'translate/', translate.translateXRight(img), '-' + str(i * 4 + 1))
        writeImage(img, fn, 'translate/', translate.translateXLeft(img), '-' + str(i * 4 + 2))
        writeImage(img, fn, 'translate/', translate.translateYDown(img), '-' + str(i * 4 + 3))
        writeImage(img, fn, 'translate/', translate.translateYUp(img), '-' + str(i * 4 + 4))


def readMultipleImages():
    i = 0
    for fn in os.listdir(Directory.CONJUNTO_PRUEBA):
        if fn.endswith(".jpg"):
            img = cv.imread(Directory.CONJUNTO_PRUEBA + fn)

            applyFilters(img, fn)

        i += 1
        if i % 20 == 0:
            print(i, ' imagenes tratadas')


def dataAugmentationBlackAndWhite():
    # dir = "/Volumes/Seagate Expansion Drive/TFT/Conjunto de datos/resized_squares"
    # images = '/Volumes/Seagate Expansion Drive/TFT/complete_dataset/'
    dir = "/Volumes/Seagate Expansion Drive/TFT/manrique/cuadros-manrique-squares"
    images = "/Volumes/Seagate Expansion Drive/TFT/manrique/cuadros-manrique-augmentation/"

    j = 0
    for img in os.listdir(dir):
        cval = 0
        background = 'black'
        if img.endswith('.jpg'):
            name_id = img[:-4]

            if not os.path.isdir(images + name_id):
                os.mkdir(images + name_id)
            img = load_img(os.path.join(dir, img))
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

            for i in range(0, 2):
                if len(fnmatch.filter(
                        os.listdir(images + name_id),
                        name_id + "_" + str(i) + "*")) >= 100:
                    if i == 1:
                        j += 1
                    continue

                if i == 1:
                    j += 1
                    background = 'white'
                    cval = 250

                datagen = ImageDataGenerator(
                    rotation_range=10,
                    fill_mode='constant',
                    cval=cval,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.2,
                    zoom_range=0.2
                )

                for batch in datagen.flow(x, batch_size=1, shuffle=True,
                                          save_to_dir=images + name_id,
                                          save_prefix=name_id + "_" + str(i) + "_" + background, save_format='jpeg'):

                    if len(fnmatch.filter(
                            os.listdir(images + name_id),
                            name_id + "_" + str(i) + "*")) >= 100:
                        break  # otherwise the generator would loop indefinitely

            if j % 50 == 0:
                print('llevo ' + str(j) + ' imágenes tratadas')
                print(datetime.datetime.now())


def dataAugmentation():
    #dir = "/Volumes/Seagate Expansion Drive/TFT/Conjunto de datos/resized_squares"
    # images = '/Volumes/Seagate Expansion Drive/TFT/complete_dataset/'
    # dir = "/Users/Yisus95/PycharmProjects/conv2d_net/data/test/test_folder"
    # images = "/Users/Yisus95/PycharmProjects/conv2d_net/preview"
    # dir = "/Volumes/Seagate Expansion Drive/TFT/manrique/cuadros-manrique-squares"
    dir = "/Volumes/Seagate Expansion Drive/TFT/Conjunto de datos/resized_squares"
    # images = "/Volumes/Seagate Expansion Drive/TFT/manrique/cuadros-manrique-augmentation"
    images = "/Volumes/Seagate Expansion Drive/TFT/"

    j = 0
    for img in os.listdir(dir):
        print(img)
        if img.endswith('.jpg'):
            name_id = img[:-4]

            if not os.path.isdir(os.path.join(images,name_id)):
                print(os.path.join(dir,name_id))
                os.mkdir(os.path.join(images,name_id))
            img = load_img(os.path.join(dir, img))
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                brightness_range=(0.8,1.5)
            )

            for batch in datagen.flow(x, batch_size=1, shuffle=True,
                                      save_to_dir=os.path.join(images,name_id),
                                      save_prefix=name_id + "_0001", save_format='jpeg'):

                print(len(os.listdir(os.path.join(images,name_id))))
                if len(os.listdir(os.path.join(images,name_id))) >= 99 :
                    break  # otherwise the generator would loop indefinitely

                break


            if j % 50 == 0:
                print('llevo ' + str(j) + ' imágenes tratadas')
                print(datetime.datetime.now())
        break


if __name__ == '__main__':
    print("aplicando filtros, ", datetime.datetime.now())
    # readMultipleImages()
    # dataAugmentation()
    # dataAugmentationBlackAndWhite()
    # print(len(val_acc))

    acc = [0.5921875, 0.9476563, 0.9800781, 0.9390625, 0.97734374, 0.99492186, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    val_acc = [0.9265625, 0.9875, 0.996875, 0.9515625, 0.9484375, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    loss = [1.2970980765298008, 0.13932063471438597, 0.07170789024858096, 0.3067086303330143, 0.10683407882243046,
     0.021132938922164612, 0.0004590917076967571, 3.496485937901639e-05, 2.2892397835860833e-05, 1.7108186425485173e-05,
     1.2064397016153805e-05, 9.140227708925863e-06, 7.523457717395487e-06, 6.280991696794303e-06, 5.224612264953521e-06]
    val_loss = [0.20390008464455606, 0.025785376063140575, 0.022141084366012365, 0.17141902623698116, 0.13455189009728202,
     0.0009847232282481854, 0.00033205134537297456, 0.00018851432559472415, 0.00010989684679714173,
     9.28815524048332e-05, 8.900508581533018e-05, 7.691324410181721e-05, 6.040125345521119e-05, 4.994696486733119e-05,
     4.9455119334140815e-05]

    import numpy as np
    fig, ax = plt.subplots()
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    ax.xaxis.set_ticks(np.arange(0, val_acc.__len__()+1, 1))
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots()
    ax.yaxis.set_ticks(np.arange(0, 4, 0.2))
    ax.xaxis.set_ticks(np.arange(0, val_loss.__len__()+1, 2))
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)
    plt.show()
    print("acabaron los filtros", datetime.datetime.now())
    sys.exit()
