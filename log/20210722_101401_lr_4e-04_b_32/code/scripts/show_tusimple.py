import json
import numpy as np
import cv2
from data.constant import tusimple_row_anchor, culane_row_anchor


COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

data_dir = '../../data/tusimple/test_set'
with open(data_dir + '/test_label.json', 'r') as f:
    lines = f.readlines()
json_list = [json.loads(line) for line in lines]
# print(len(json_list))
for ele in json_list:
    # print(ele)
    lanes = np.asarray(ele['lanes'])
    # tran_set
    # h_samples = np.asarray(ele['h_samples'])
    # test tusimple
    h_samples = np.array(tusimple_row_anchor)
    print('h_samples', h_samples)
    raw_file = ele['raw_file']
    img = cv2.imread(data_dir + '/' + raw_file)

    # for i in range()
    # print(h_samples.dtype, h_samples.shape, lanes.dtype, lanes.shape)
    for j in range(lanes.shape[0]):
        lane = lanes[j]
        color = COLORS[j]
        for i in range(h_samples.shape[0]):

            pt = (lane[i], h_samples[i])
            print(pt)
            cv2.circle(img, pt, 2, color, 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
