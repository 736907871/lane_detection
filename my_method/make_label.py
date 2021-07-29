import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import numpy as np
import cv2
import heapq
import random
import math
from binary_line import binary
from erode_image import erode
import sys
import json
import os

#use json  which has all line information to make every line image with label.And make every line json.
# def make():
#     # input_file=open(json_path, 'r' , encoding='utf-8')
#     #
#     # json_decode=json.load(input_file)
#     # print(json_decode)
#     if not os.path.exists('/home/alex/data_image/json/'):
#         os.makedirs('/home/alex/data_image/json/')
#
#     with open("/home/alex/data_image/lane.json", "r") as f:
#         lines = f.readlines()
#         # print(len(lines))
#         for line in lines:
#             dict = json.loads(line)
#             # print(dict)
#             lanes = dict['lanes']
#             h = dict['h_samples']
#             path = dict['raw_file']
#             image_name = path.split('/')[-1]
#             image = cv2.imread('/home/alex/data_image/image/'+image_name)
#
#             line_18=[]
#             for i,lane in enumerate(lanes):
#                 l=[]
#                 for j,x in enumerate(lane):
#                     point = [h[j],x]
#                     l.append(point)
#                 line_18.append(l)
#
#             for i, line in enumerate(line_18):
#                 # print(len(line))
#                 # print(line)
#                 for coor in line:
#                     if coor[1] < 0: continue
#                     # print(coor)
#                     r = 50 + i * 50
#                     g = 50 + i * 50  # 50+i*50
#                     b = 50 + i * 50
#                     cv2.circle(image, (int(coor[1]), int(coor[0])), 2, r, 16)
#
#             cv2.imwrite('/home/alex/data_image/image/'+image_name, image)
#
#             image_name = path.split('/')[-1]
#             image_name = image_name.split('.')[0]
#             with open('/home/alex/data_image/json/' + image_name+'.json', "w", encoding='utf-8') as f2:  # 采用utf-8
#                 json.dump(dict, f2)

#make all image to resize to (1280,720) ,and save them.
path = '/home/alex/zwh/lane_marking_examples/' # 数据集路径
savePath = '/home/alex/zwh/imgdata_1280X720'
# 循环遍历lfw数据集下的所有子文件
for road_file in os.listdir(path):
    label_file = path + road_file + '/ColorImage/'
    for record_file in os.listdir(label_file):
        for camera_file in os.listdir(label_file + record_file + '/'):
            for im_file in os.listdir(label_file + record_file + '/' + camera_file + '/'):
                img_path = label_file + record_file + '/' + camera_file + '/'+im_file
                tt=cv2.imread(img_path)
                tt= cv2.resize(tt,(1280,720))
                imgsavePath = savePath + "/" + im_file
                print(imgsavePath)
                cv2.imwrite(imgsavePath,tt)
