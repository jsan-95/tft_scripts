import fnmatch

from directoryPath import Directory
import cv2 as cv
import os
from matplotlib import pyplot as plt
import datetime
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



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
        cv.imwrite(folder_dst, dst)


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
    images = '/Volumes/Seagate Expansion Drive/TFT/complete_dataset/'
    dir = "/Users/Yisus95/PycharmProjects/conv2d_net/data/test/test_folder"

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

            if j % 50 == 0:
                print('llevo ' + str(j) + ' imágenes tratadas')
                print(datetime.datetime.now())

if __name__ == '__main__':
    print("aplicando filtros, ", datetime.datetime.now())
    # dataAugmentation()
    dataAugmentationBlackAndWhite()

    print("acabando filtros, ", datetime.datetime.now())
