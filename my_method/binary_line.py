# -*- coding:utf-8 -*-

import cv2
import numpy as np

def binary(path):
    im = cv2.imread(path)
    img = cv2.resize(im, (1280, 720))
    # cv2.imshow('img', im)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow('',gray)
    cv2.waitKey(0)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ret,gray1 = cv2.threshold(gray,137,255,cv2.THRESH_TOZERO)
    ret,gray1 = cv2.threshold(gray1,139, 255, cv2.THRESH_TOZERO_INV)
    ret,gray2 = cv2.threshold(gray,54,255,cv2.THRESH_TOZERO)
    ret,gray2 = cv2.threshold(gray2,56, 255, cv2.THRESH_TOZERO_INV)
    ret,gray3 = cv2.threshold(gray,41,255,cv2.THRESH_TOZERO)
    ret,gray3 = cv2.threshold(gray3,43, 255, cv2.THRESH_TOZERO_INV)
    ret ,gray4 = cv2.threshold(gray,28,255,cv2.THRESH_TOZERO)
    ret, gray4 = cv2.threshold(gray4, 31, 255, cv2.THRESH_TOZERO_INV)

    img_add1 = cv2.addWeighted(gray1, 1, gray2, 1, 0)
    img_add2 = cv2.addWeighted(gray3, 1, gray4, 1, 0)
    img_add = cv2.addWeighted(img_add1, 1, img_add2, 1, 0)
    ret, img_binary = cv2.threshold(img_add, 28, 255, cv2.THRESH_BINARY)

    # cv2.imshow('img',img_binary)
    # cv2.imwrite('./im_binary.png',img_binary)
    # cv2.waitKey(0)
    return img_binary,img

# if __name__ == '__main__':
    # binary('/home/alex/zwh/data/Labels_road03/Label/Record014/Camera 5/171206_031730447_Camera_5_bin.png')
    # img2 = img.convert('RGB')
    # pixdata = img2.load()
    # for y in range(img2.size[1]):
    #     for x in range(img2.size[0]):
    #         if pixdata[x, y][0] == 0 and pixdata[x, y][1] == 0 and pixdata[x, y][2] < 256:
    #             pixdata[x, y] = (255, 255, 255, 0)

    # img2.show()
    # height,width,channels = img.shape
    # for j in range(width):
    #     for i in range(height):
    #         if img[i,j][0] != 180 or img[i,j][1] != 173 or img[i,j][2] != 43:
    #             img[i,j] = [0,0,0]

    # cv2.namedWindow("img")
    # cv2.setMouseCallback("img", mouse_click)
    # while True:
    #     cv2.imshow('img', img)
    #     if cv2.waitKey() == ord('q'):
    #         break
    # cv2.destroyAllWindows()
