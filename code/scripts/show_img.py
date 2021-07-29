import cv2
import numpy as np
img_path = '../../data/tusimple/train_set/clips/0531/1492635078640922645/20.png'
img = cv2.imread(img_path, flags=-1)
img[img == 1] = 255
print(img.shape, img.dtype, np.sum(img.reshape(-1,)), id(img))
# img = img[:1920, :1920]
# print(img.shape, img.dtype, np.sum(img.reshape(-1,)), id(img))
cv2.imshow('img', img)
cv2.waitKey(0)
