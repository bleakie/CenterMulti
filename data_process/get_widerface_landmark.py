# -*- coding:utf-8 -*-

import json
import cv2
import os
import numpy as np
import re

def get_landmark(txt_path, save_path):
    txt_write = open(save_path, 'w')
    annotationfile = open(txt_path)
    min_bbox = 10
    blur_value = 0.3
    Init = False
    img_path = None
    bbox_landmarks = []
    while (True):
        line = annotationfile.readline()[:-1]
        if not line:
            break
        if re.search('jpg', line):
            if Init:
                if len(bbox_landmarks)<1:
                    continue
                txt_write.write(img_path + '\n')
                txt_write.write(str(len(bbox_landmarks)) + '\n')
                for lm in bbox_landmarks:
                    txt_write.write(str(lm)+'\n')
            Init = True
            img_path = line.split('# ')[1]#line[2:]
            bbox_landmarks = []
            continue
        else:
            values = line.strip().split()
            bbox = values[:4]
            if min(int(bbox[2]), int(bbox[3])) < min_bbox:
                continue
            if len(values) > 4:
                if float(values[19]) < blur_value:
                    continue
                for li in range(5):
                    value = float(values[(li+2)*3])
                    if value == 0:  # visible
                        values[(li + 2) * 3] = str(2)
                    elif value == 1:  # visible
                        values[(li + 2) * 3] = str(1)
                    else:
                        values[3*li+4] = str(0)
                        values[3*li+5] = str(0)
                        values[3*li+6] = str(0)
            values = ' '.join(values)
            bbox_landmarks.append(values)
    txt_write.close()
    annotationfile.close()

if __name__ == '__main__':
    txt_path = 'label.txt'
    save_path = 'val.txt'
    get_landmark(txt_path, save_path)
