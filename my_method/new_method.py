# from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import heapq
import random
import math
from binary_line import binary
from erode_image import erode
# from make_label import make
import sys
import json
import os

point_size = 1
thickness = 4  # 0 、4、8

image_height = 2710
image_width = 3384
#38 y value
section = [200, 214, 228, 242, 256, 270, 284, 298, 312, 326, 340, 354, 368, 382, 396, 410, 424, 438, 452, 466, 480, 494, 508, 522, 536, 550, 564, 578, 592, 606, 620, 634, 648, 662, 676, 690, 704, 718]
percetive_sective = [472, 475, 477, 480, 483, 486, 488, 491, 494, 497, 500, 503, 507, 510, 513, 517, 520, 524, 527, 531, 535, 539, 543, 547, 551, 555, 559, 564, 568, 573, 578, 583, 588, 593, 598, 603, 609, 615]

up = 200#/720*image_height
down = 720#image_height+1
left = 70#/1280*image_width
right = 1200#/1280*image_width
delate = 36
line_width_max = 70
midle = 1280/2

up_percetive = 472
down_percetive = 654

roi_corners = [(320, 440), (940, 440), (1050, 616), (0, 616)]
dst_corners = np.float32([[(0, 0), (1280, 0), (1280, 720), (0, 720)]])

def y_B2S(line):#y from Big to small
    line.sort(key = lambda x:x[0],reverse=True)

def y_S2B(line):
    line.sort(key=lambda x: x[0], reverse=False)

def lines_B2S(lines):
    for line in lines:
        y_B2S(line)
# 选择ROI区域 只保留此区域 其他区域用255填充。
def roi_mask(img, vertices):
  #定义mask全为黑
  mask = np.zeros_like(img)
  #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
  if len(img.shape) > 2:
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    mask_color = (255,) * channel_count
  else:
    mask_color = 255
  #将区域和图片进行填充fillPoly和叠加and
  cv2.fillPoly(mask, vertices, mask_color)
  masked_img = cv2.bitwise_and(img, mask)
  return masked_img

# 3.# 将img转换成鸟瞰图 透视变换(Perspective Transformation)
def perspective_transform(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def image_perspective(im):
    # 1读取图片 1280*720
    img = im  # cv2.imread(path,0)
    # cv2.imshow('img1', img)

    # 2、选择ROI区域
    roi_vtx = np.array([roi_corners])
    # print(roi_vtx)

    roi_mask_img = roi_mask(img, roi_vtx)
    # cv2.imshow('img',roi_mask_img)
    wrap_offset = 250

    # 原图像中能够表示一个矩形的四个点的坐标 为实际图像中的车道线选框点
    src_corners = roi_corners

    M = cv2.getPerspectiveTransform(np.float32(src_corners), np.float32(dst_corners))
    # 透视变换(Perspective Transformation)
    wrap_img = perspective_transform(roi_mask_img, M)
    # cv2.imshow('11',wrap_img)
    wrap_img = cv2.cvtColor(wrap_img, cv2.COLOR_GRAY2RGB)
    # cv2.imshow('',wrap_img)
    # cv2.waitKey(0)
    return wrap_img

# cancel two lines the same points
def cancel_same_points(all_line_sort):
    for i, line in enumerate(all_line_sort):
        # print(line)
        if len(all_line_sort[i]) < 2: continue
        j = i + 1
        while j < len(all_line_sort):
            if len(all_line_sort[j]) < 2:
                j += 1
                continue
            k = -1
            while abs(k) <= len(all_line_sort[i]) and abs(k) <= len(all_line_sort[j]) and all_line_sort[i][k] == \
                    all_line_sort[j][k]:
                k -= 1
            # print(j,k)
            # print(len(all_line_sort[3]))
            if k != -1 and abs(k) <= len(all_line_sort[i]) and abs(k) <= len(all_line_sort[j]):
                p1 = all_line_sort[i][k]
                p2 = all_line_sort[j][k]
                p = all_line_sort[i][k + 1]
                x1 = abs(p1[1] - p[1])
                x2 = abs(p2[1] - p[1])
                k += 1
                k = abs(k)  # same points number are k
                if x1 <= x2:
                    while k > 0:
                        all_line_sort[j].pop(-1)
                        k -= 1
                else:
                    while k > 0:
                        all_line_sort[i].pop(-1)
                        k -= 1
            j += 1

def create_line(all_line,pts):
    # print('create')
    # print('pts',pts)
    # print('all_line',all_line)
    for p in pts:
        b=False
        for line in all_line:
            if p in line:
               b=True
               break
        if b==False:
            all_line.append([p])

#points is from B to S(y'value)
def create_points(wrap_img):
    line=[]

    for i in range(up,down,(down-up)//delate):
        j = left
        while j < right:
            if wrap_img[i][j][0]==255:
                num = 1
                value_x = j
                j = j+1
                for t in range(line_width_max):
                    if j>=right:break
                    if wrap_img[i][j][0] == 255:
                        num += 1
                        value_x += j
                    j=j+1
                line.append([i,int(value_x//num)])
            else:j = j+1

    lines_pts=[]
    for i in range(up,down,(down-up)//delate):
        l=[]
        for pts in line:
            if pts[0]==i:
                l.append(pts)
        lines_pts.append(l)

    lines_pts = list(reversed(lines_pts))#y from b to s
    return lines_pts

#from lines_pts to create lines(kernal algrizorm)
def generate_line(lines_pts):
    all_line = []

    for i in range(delate):  # only need delate y value
        # print(i, all_line)
        # len_allline = len(all_line)
        # for j in range(len_allline):
        seg_line(lines_pts, all_line, i)
        # print(i, all_line)

        create_line(all_line, lines_pts[i])

        # print(i,all_line)
        for line in all_line:
            find_min_dis_point(line, lines_pts, i)
        # print(i, all_line)
    all_line.sort(key=lambda x: len(x), reverse=True)
    # print(2222)
    all_line_sort = all_line[0:4]
    return all_line_sort

def find_min_dis_point(lane,line_pts,i):#return y from B to S
    start_point = lane[-1]
    # if len(line) < 2: k1 = None
    # else:
    #     second_point = line[-2]
    #     k1 = find_k(start_point,second_point)
    dis=sys.float_info.max
    temp=[]
    i +=1
    b=True
    pts = line_pts[i]
    while i<len((line_pts)):
        pts = line_pts[i]
        for p in pts:
            if abs(p[1] - start_point[1]) < line_width_max+10:
                b = False
                temp = p
                break
        if b==False:break
        i += 1
    # while i<len(line_pts) and b:
    #     pts = line_pts[i]
    #     for p in pts:
    #     # k2 = find_k(start_point,p)
    #     # if k1!=None:k=abs(k1-k2)
    #         d=math.sqrt((p[1]-start_point[1])*(p[1]-start_point[1])*1000+(p[0]-start_point[0])*(p[0]-start_point[0]))
    #         if d<dis and abs(p[1]-start_point[1])<line_width_max+10: #and (k1==None or k<10):
    #             dis=d
    #             temp=p
    #     i+=1
    if len(temp)!=0:
        if temp not in lane:
            lane.append(temp)

    y_B2S(lane)

def find_k(p1,p2):
    # angle_line = (p1[1]* p2[1] + p1[0] * p2[0]) / math.sqrt((p1[1] * p1[1]+ p1[0] * p1[0] ) * (p2[1] * p2[1] + p2[0] * p2[0]) + 1e-10)
    if (p1[1]-p2[1])==0:return 90
    k = math.atan((p1[0]-p2[0])/(p1[1]-p2[1]))*180.0/3.14
    if k >0 :
        return k
    # print(angle_line)
    else :return 180+k

def seg_line(line_pts,all_line,i):
    for lane in all_line:
        find_min_dis_point(lane,line_pts,i)
    #print(line)
    for line in all_line:
        if(len(line)>2):
            #print(line[-2],line[-3])
            k1 = find_k(line[-1],line[-2])
            k2 = find_k(line[-2],line[-3])

            #print(k1,k2)
            if abs(k1-k2)>90:
                p1 = line[-1]
                p2 = line[-2]
                # same_lines_1 = [line for line in all_line if line[-1]==p1]
                # same_lines_2 = [line for line in all_line if line[-1]==p1 and len(line)>2 and line[-2]==p2]
                # if len(same_lines_2)>1:#len(same_lines_1)==1 or
                #     for same_line in same_lines_2:
                p1=line.pop(-1)
                #print(line)
                p2=line.pop(-1)
                if len(line)==1:
                    all_line.append([p2,p1])

def construct_line(line):# for point in middle
    #line y from big to small
    # line_reversed = line#list(reversed(line))
    if len(line)<2:return line
    y_B2S(line)
    lane = []
    start_y = line[0][0]#big_y
    end_y = line[len(line)-1][0]#small_y
    t = 0
    points = []
    for i in range(start_y,end_y-(down-up)//delate,-((down-up)//delate)):
        points.append([i,-2])#y from big to small
    for p in line:
        for point in points:
            if p[0] == point[0]:
                point[1] = p[1]

    for i,point in enumerate(points):
        if points[i][1] == -2:
            j = i+1
            while j < len(points) and points[j][1] == -2:
                j += 1
            if j < len(points):
                if points[j][1]-points[i-1][1] == 0:
                    points[i][1] = points[i-1][1]
                else:
                    k = (points[j][0] - points[i - 1][0]) / (points[j][1] - points[i - 1][1])
                    b = points[j][0] - k * (points[j][1])
                    if k != 0:
                        points[i][1] = int((points[i][0] - b) // k)
            else:
                break
    # for i,de in enumerate(range(start_y,end_y,-((down-up)//delate))):
    #     # print(i,de)
    #     if line[t][0] != de:
    #         # print(line[t][0],de)
    #         if line[t][1] - line[t-1][1] == 0:
    #             lane.insert(i, [de, line[t][1]])
    #         else :
    #             k = (line[t][0] - line[t-1][0])/(line[t][1] - line[t-1][1])
    #             b = line[t][0] - k*(line[t][1])
    #             if k!=0:
    #                 lane.insert(i,[de,int((de-b)//k)])#/home/alex/zwh/lane_marking_examples/road03/Label/Record003/Camera 5/171206_030143355_Camera_5_bin.png#k==0
    #             else:
    #                 # print(111)
    #                 line.pop(t)
    #         # t += 2
    #     else:
    #         lane.insert(i, [de, line[t][1]])
    #         t += 1
    # line_reversed = line_reversed[0,len(line_reversed),]
    # lane.insert(len(lane)-1,[line[len(line)-1][0], line[len(line)-1][1]])
    # line_reversed = list(reversed(lane))
    # print(len(line_reversed))
    return points

def complition_line(line):# make some point not exists to becoome x=-2
    p_s = []
    for y in percetive_sective:
        p_s.append([y,-2])
    for p in line:
        for q in p_s:
            if p[0] == q[0]:
                q[1] = p[1]
                break
    return p_s

def reversed_xy(line):
    for i,p in enumerate(line):
        line[i] = list(reversed(p))

def kernal(im):
    wrap_img = image_perspective(im)

    lines_pts = create_points(wrap_img)

    all_line_sort = generate_line(lines_pts)

    lines_B2S(all_line_sort)#y from big to small

    cancel_same_points(all_line_sort)


    all_line_construct=[]
    for line in all_line_sort:
        constr_line = construct_line(line)
        if len(constr_line)>10:
            all_line_construct.append(constr_line)
    # print(all_line_construct)
    all_line_construct.sort(key=lambda x: x[0][1],reverse=False)

    #make picture with points
    # for i,line in enumerate(all_line_construct):
    #     # print(len(line))
    #     # if i==7 :
    #     r = 50+i*50
    #     g = 50+i*50#50+i*50
    #     b = 50+i*50
    #     for coor in line:
    #         cv2.circle(wrap_img, (int(coor[1]), int(coor[0])), point_size, [r,g,b], thickness)
    # cv2.imshow('aaa',wrap_img)
    # cv2.waitKey(0)

    src_corners = roi_corners
    M = cv2.getPerspectiveTransform(np.float32(dst_corners), np.float32(src_corners))
    all_line_percetive=[]

    #first we make all line return percetive then complition line
    for line in all_line_construct:
        # print(all_line_construct[1])
        reversed_xy(line)
        points = np.asarray(line)
        points = points.reshape(1, -1, 2).astype(np.float32)  # 二维变三维， 整形转float型， 一个都不能少

        new_points = cv2.perspectiveTransform(points,M)
        new_points = new_points.reshape((-1,2)).astype(np.int)
        new_points = new_points.tolist()
        # print(new_points)
        reversed_xy(new_points)

        new_points = complition_line(new_points)
        # print(new_points)

        all_line_percetive.append(new_points)

    wrap_img = perspective_transform(wrap_img, M)
    # cv2.imshow('',wrap_img)
    return wrap_img,all_line_percetive

def process_lines(all_line):# return 18 number point
    all_line_18=[]
    for line in all_line:
        # print(len(line))
        l=[]
        for i in range(0,len(line)-(len(line))//18,(len(line))//18):
            x = int(line[i][1])#*image_width/1280)
            y = int(line[i][0])#*image_height/720)
            l.append([y,x])
        # print(len(l))
        all_line_18.append(l)
    return all_line_18

def back_size(all_line):#resize to when it begin'size
    # print(len(all_line))
    # print(all_line)
    for line in all_line:
        for i in range(len(line)):
            if line[i][1]==-2:
                line[i][0] = int(float(line[i][0])*float(image_height)/float(720))
            else:
                line[i][1] = int(float(line[i][1])*float(image_width)/float(1280))
                line[i][0] = int(float(line[i][0])*float(image_height)/float(720))
    # print(len(all_line))
    # print(all_line)
    return all_line

def first_point_x(line):
    i = 0
    y_B2S(line)
    # print(lane)
    while i<len(line):
        if line[i][1] == -2:
            i+=1
        else:
            # print(lane[i][1])
            return line[i][1]
    return -2

def class_left2right(lines):#use middle=1280/2,so the lines are the lane from 1280*720 image
    lines.sort(key=lambda x:first_point_x(x),reverse=False)

    all_line=[[],[],[],[]]

    for i,line in enumerate(lines):
        if first_point_x(line)<=midle:
            all_line[0] = all_line[1]
            all_line[1] = line
        else:
            if len(all_line[2]) == 0:
                all_line[2] = line
            elif len(all_line[3]) == 0:
                all_line[3] = line
            else:pass
    reverse_per = list(reversed(percetive_sective))
    for line in all_line:
        if len(line)==0:
            for i in reverse_per:
                line.append([i,-2])
    return all_line

def convert2json(path_image):#path is oraginal image
    print(path_image)
    # # path = '/home/alex/zwh/data/Labels_road02/Label/Record001/Camera 5/170927_063944851_Camera_6_bin.png'
    image_path_part = path_image.split('Label/')[1]#Record001/Camera 5/170927_063944851_Camera_6_bin.png
    image_name = image_path_part.split('/')[-1]
    image_name_part = image_name.split('.')[0]
    colormap_path = '/home/alex/zwh/data/ColorImage_road03/ColorImage/'+image_path_part.split('_bin')[0]+'.jpg'
    colormap = cv2.imread(colormap_path)
    colormap_resize = cv2.resize(colormap,(1280,720))

    binary_image,previous_img = binary(path_image)
    erode_image = erode(binary_image)
    img, all_line = kernal(erode_image)#return orginal's image size lines
    #after kernal() y from S to B
    all_line_sequence = class_left2right(all_line)
    #after class_left2right() y from B to S
    lines_18 = process_lines(all_line_sequence)
    #resize_image_line = return_size(lines_18)
    # print(len(erode_image))
    # for i,line in enumerate(lines_18):
    #     # print(len(line))
    #     # print(line)
    #     for coor in line:
    #         if coor[1]<0:continue
    #         # print(coor)
    #         r = 50 + i * 50
    #         g = 50 + i * 50  # 50+i*50
    #         b = 50 + i * 50
    #         cv2.circle(image, (int(coor[1]), int(coor[0])), 2, [r,g,b], 4)
    # if not os.path.exists('/home/alex/data_image/image/'):
    #     os.makedirs('/home/alex/data_image/image/')
    # cv2.imwrite('/home/alex/data_image/image/'+image_name, image)

    # cv2.imshow('',cv2.resize(image,(1280,720)))
    # cv2.waitKey(0)

    # back_size(lines_18)
    for line in lines_18:
        line.sort(key = lambda x:x[0],reverse=False)
    # print(lines_18)
    line_x = []
    line_y = []
    for p in lines_18[0]:
        line_y.append(p[0])
    for line in lines_18:
        l = []
        for p in line:
            l.append(p[1])
        line_x.append(l)
    dict = {"lanes": line_x, "h_samples": line_y,"raw_file":colormap_path}

    for i, line in enumerate(lines_18):
        # print(len(line))
        # print(line)
        for coor in line:
            if coor[1] < 0: continue
            # print(coor)
            r = 50 + i * 50
            # g = 50 + i * 50  # 50+i*50
            # b = 50 + i * 50
            cv2.circle(previous_img, (int(coor[1]), int(coor[0])), 2, (0,0,r), 16)#BGR

    cv2.imwrite('/home/alex/zwh/data_process/728/label_728/' + image_name, previous_img)
    cv2.imwrite('/home/alex/zwh/data_process/728/image_728/' + image_name_part+'.jpg', colormap_resize)

    jsonName = '/home/alex/zwh/data_process/728/json_728/' + image_name_part + '.json'
    with open(jsonName, "w", encoding='utf-8') as f2:  # 采用utf-8
        json.dump(dict, f2)

    with open("/home/alex/zwh/data_process/728/lane.json", "a+", encoding='utf-8') as f:  # 采用utf-8
        json.dump(dict, f)
        f.write('\n')
    # cv2.imshow('', img)

if __name__ == "__main__":

    path = '/home/alex/zwh/data/Labels_road03/Label/' # 数据集路径
    # 循环遍历lfw数据集下的所有子文件
    # for road_file in os.listdir(path):
    #     label_file = path + road_file + '/Label/'
    for record_file in os.listdir(path):
        for camera_file in os.listdir(path + record_file + '/'):
            for im_file in os.listdir(path + record_file + '/' + camera_file + '/'):
                # convert2json(path + record_file + '/' + camera_file + '/' + im_file)
                try:
                    convert2json(path + record_file + '/' + camera_file + '/'+im_file)
                except Exception as e:
                    with open("error.txt", "a") as f:
                        f.write(path + record_file + '/' + camera_file + '/'+im_file+'\n')
    # convert2json('/home/alex/zwh/data/Labels_road03/Label/Record032/Camera 5/171206_033949996_Camera_5_bin.png')