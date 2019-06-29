import sys

PY3 = sys.version_info[0] == 3
if PY3:
    xrange = range

DIRECTORY_DISK = "/Volumes/Seagate Expansion Drive/TFT/Conjunto de datos/"

import numpy as np
import cv2 as cv


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

def isAreaMax(cnt):
    if cnt[0][0] < 40 and cnt[0][1] < 40:
        return True
    return False

def isCentered(cnt, img):
    M = cv.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    width, height = img.shape[:2]
    width /= 2
    height /= 2

    if abs(width - cY) < 100 and abs(height - cX) < 100:
        return True
    return False



def cleanContour(cnt):
    for i in range(len(cnt)):
        x0, y0 = cnt[i][0], cnt[i][1]
        for j in range(i, len(cnt)):
            x1, y1 = cnt[j][0], cnt[j][1]
            diff_x = x0 - x1
            diff_y = y0 - y1

            if abs(diff_x) < 100:
                if diff_x > 0:
                    cnt[j][0] = x0
                else:
                    cnt[i][0] = x1

            if abs(diff_y) < 100:
                if diff_y > 0:
                    cnt[j][1] = y0
                else:
                    cnt[i][1] = y1
    return cnt


def find_squares(img):
    img = cv.GaussianBlur(img, (11, 11), 0)

    squares = []

    cv.imwrite('/Users/Yisus95/PycharmProjects/conv2d_net/gaussian_blur1.jpg', img)

    j = 0
    for gray in cv.split(img): # divide rgb
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 10, apertureSize=5)
                bin = cv.dilate(bin, None)
                j += 1
                cv.imwrite('/Users/Yisus95/PycharmProjects/conv2d_net/contour_'+str(j)+'.jpg', bin)
            else:
                _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            bin, contours, _hierarchy = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)


            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.015 * cnt_len, True)
                if len(cnt) == 4 and cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):

                    if len(squares) > 0 and cv.contourArea(cnt) > cv.contourArea(squares[0]):
                        continue
                    if not isCentered(cnt, img):
                        continue

                    cnt = cnt.reshape(-1, 2)
                    # max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in xrange(4)])
                    # if max_cos < 0.2:
                    if isAreaMax(cnt):
                       continue
                    if len(squares) > 0 and cv.contourArea(cnt) > 1000:
                        squares.pop()
                    if cv.contourArea(cnt) > 1000:
                        squares.append(cnt)

    return squares


def cropImage(squares, img):
    points = np.array(squares[0])
    res = np.array(sorted(points, key=getKey))

    p0 = 0
    p1 = 1
    p2 = 2
    p3 = 3

    # Corte de los puntos
    # izq inf, der inf, izq sup, der sup
    if res[0][1] > res[1][1]:
        p0 = 1
        p1 = 0
    if res[2][1] > res[3][1]:
        p2 = 3
        p3 = 2

    pts1 = np.float32([res[p0], res[p2], res[p1], res[p3]])
    pts2 = np.float32([[0, 0], [758, 0], [0, 825], [758, 825]])

    M = cv.getPerspectiveTransform(pts1, pts2)
    img = cv.warpPerspective(img, M, (758, 825))

    return img

def getKey(item):
    return item[0]

def readOneImage():

    img = cv.imread("/Volumes/Seagate Expansion Drive/TFT/Conjunto de datos/public_images/11506.jpg")
    img = cv.resize(img, (758, 825))

    squares = find_squares(img)

    # cv.drawContours(img, np.array(squares), -1, (0, 255, 0), 3)

    print("pasoo")
    if len(squares) > 0:
        img = cropImage(squares, img)

    print("pasoo")
    cv.imwrite('/Users/Yisus95/PycharmProjects/conv2d_net/prueba.jpg', img)


def readMultipleImages():
    i = 0
    directory_manrique = "/Volumes/Seagate Expansion Drive/TFT/manrique/cuadros-manrique-resized-javier/"
    # for fn in os.listdir(DIRECTORY_DISK + "resized_images/"):
    for fn in os.listdir(directory_manrique):
        if fn.endswith(".jpg"):
            # img = cv.imread(DIRECTORY_DISK + "resized_images/" + fn)
            img = cv.imread(directory_manrique + fn)
            squares = find_squares(img)
            if len(squares) == 0:
                # cv.imwrite(DIRECTORY_DISK + 'others/' + fn, img)
                # cv.imwrite("/Volumes/Seagate Expansion Drive/TFT/manrique/cuadros-manrique-squares-javier/" + fn, img)
                cv.imwrite("/Users/Yisus95/PycharmProjects/conv2d_net/manrique/" + fn, img)
            else:
                points = np.array(squares[0])
                res = np.array(sorted(points, key=getKey))

                p0 = 0
                p1 = 1
                p2 = 2
                p3 = 3

                if res[0][1] > res[1][1]:
                    p0 = 1
                    p1 = 0
                if res[2][1] > res[3][1]:
                    p2 = 3
                    p3 = 2

                pts1 = np.float32([res[p0], res[p2], res[p1], res[p3]])
                pts2 = np.float32([[0, 0], [758, 0], [0, 825], [758, 825]])

                M = cv.getPerspectiveTransform(pts1, pts2)
                img = cv.warpPerspective(img, M, (758, 825))
                # cv.imwrite("/Volumes/Seagate Expansion Drive/TFT/manrique/cuadros-manrique-squares-javier/" + fn, img)
                cv.imwrite("/Users/Yisus95/PycharmProjects/conv2d_net/manrique/" + fn, img)
        if i == 100:
            print("iteración: ", i)
        i += 1


if __name__ == '__main__':
    import os
    import datetime

    print("Que empiecen los juegos del hambre: ", datetime.datetime.now())
    readOneImage()
    # readMultipleImages()

    print("Katniss ganadora de los juegos: ", datetime.datetime.now())
