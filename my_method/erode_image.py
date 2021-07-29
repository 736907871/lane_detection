# coding=utf-8
import cv2
import numpy as np
def erode(im):
    img = im#cv2.imread(path, 0)
    kernel = np.ones((2, 2), dtype=np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    ss = np.hstack((img, erosion))
    #cv2.imshow('erode',ss)

    kernel = np.ones((3, 3), dtype=np.uint8)
    dilate = cv2.dilate(erosion, kernel, 1) # 1:迭代次数，也就是执行几次膨胀操作
    # cv2.imshow('dilate',dilate)
    # cv2.imwrite('./tt.png',dilate)
    # cv2.waitKey(0)

    return dilate
# 定义结构元素
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# # 开运算
# #opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# opened = cv2.morphologyEx(img ,cv2.MORPH_OPEN, kernel)
#
# cv2.imshow("Open", opened);

# kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
# # 闭运算
# closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel2)

# 显示腐蚀后的图像
#cv2.imshow("Close", closed);

#cv2.imwrite("erode_image.png",closed)



# minLineLength = 5
# maxLineGap = 12
# lines = cv2.HoughLinesP(dilate, 1, np.pi / 180, 10, minLineLength, maxLineGap)
# for i in range(len(lines)):
#     for x1, y1, x2, y2 in lines[i]:
#         dilate = cv2.line(dilate, (x1, y1), (x2, y2), i, 2)
#
#
# cv2.imshow("houghline",dilate)
# cv2.waitKey()
# cv2.destroyAllWindows()

# thresh=dilate
# contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#得到轮廓信息
# cnts=contours[:]

# print(len(cnts))
# for cnt in cnts:
    # rows, cols = thresh.shape[:2]
    # print(rows, cols)
    # [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    # lefty = int((-x * vy / vx) + y)
    # righty = int(((cols - x) * vy / vx) + y)
    # cv2.line(thresh, (cols - 1, righty), (0, lefty), 100, 2)


# thresh = cv2.drawContours(thresh,contours,-1,100,1)#thresh is img with bounding box (pixel'value is 100)
# cv2.imshow('thresh',thresh)

# for cnt in cnts:
#     x, y, w, h = cv2.boundingRect(cnt)
#
#     # Rotate Rectangle
#     rect = cv2.minAreaRect(cnt)
#     box = cv2.boxPoints(rect)
#     box = np.int64(box)
#     img = cv2.drawContours(thresh, [box], -1, 100, 1)

# cv2.imshow('boundingRect', thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
