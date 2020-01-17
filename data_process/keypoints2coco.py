# -*- coding:utf-8 -*-

"""只需要按照实际改写images，annotations，categories另外两个字段其实可以忽略
在keypoints，categories内容是固定的不需修改
"""

import json
from tqdm import tqdm
import cv2
import os
import numpy as np
import re

class COCO(object):
    def info(self):
        return {"version":"1.0",
                "year":2020,
                "contributor":"Mr.yang",
                "date_created":"2018/08/21",
                "github":"https://github.com/bleakie"}
    def licenses(self):
        return [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "name": "Attribution-NonCommercial-ShareAlike License",
            "id": 1
        }
    ]
    def image(self):
        return {
            "license": 4,
            "file_name": "000000397133.jpg", # 图片名
            "coco_url":  "http://images.cocodataset.org/val2017/000000397133.jpg",# 网路地址路径
            "height": 427, # 高
            "width": 640, # 宽
            "date_captured": "2013-11-14 17:02:52", # 数据获取日期
            "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",# flickr网路地址
            "id": 397133 # 图片的ID编号（每张图片ID是唯一的）
        }

    def annotation(self):
        return {
            "segmentation": [ # 对象的边界点（边界多边形）
                [
                    0.,0.,# 第一个点 x,y坐标
                    0.,0., # 第二个点 x,y坐标
                    0.,0.,
                    0.,0.
                ]
            ],
            "num_keypoints": 5,
            # keypoints是按照以下关键点来标记的，如果nose 没有标则为0,0,0(3个数字为一组，分别为x,y,v v=0表示为标记此时的x=y=0,
            # v=1表示标记了但是在图上是不可见，v=2表示标记了，在图上可见)
            # "nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow",
            # "right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"
            "keypoints": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "area": 0., # 区域面积
            "iscrowd": 0, #
            "image_id": 397133, # 对应的图片ID（与images中的ID对应）
            "bbox": [0.,0.,0.,0.], # 定位边框 [x,y,w,h]
            "category_id": 1, # 类别ID（与categories中的ID对应）
            "id": 82445 # 对象ID，因为每一个图像有不止一个对象，所以要对每一个对象编号（每个对象的ID是唯一的）
            }

    def categorie(self):
        return {
                "supercategory": "face", # 主类别
                "id": 1, # 类对应的id （0 默认为背景）
                "name": "face", # 子类别
                "keypoints": ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"],
                # "skeleton": [[1, 3], [2, 3], [3, 4], [3, 5]]
                }

class Keypoints2COCO(COCO):
    def __init__(self,txt_path,save_json_path,images_path):
        self.data = open(txt_path)
        self.save_json_path = save_json_path # 最终保存的json文件
        self.images_path=images_path # 原始图片保存的位置
        self.images = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.num=1
        self.keypoints=["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]
        self.num_keypoints=5

    def __call__(self):
        while (True):
            img_path = self.data.readline()[:-1]
            if not img_path:
                break
            if re.search('jpg', img_path):
                if not os.path.exists(os.path.join(self.images_path, img_path)):
                    continue
                # init image
                image=self.image()
                image["file_name"] = img_path
                image["id"] = self.num
                img = cv2.imread(os.path.join(self.images_path, img_path))
                if img is None:
                    continue
                image["height"] = img.shape[0]
                image["width"] = img.shape[1]

                line = self.data.readline()[:-1]
                if not line:
                    break
                facenum = (int)(line)
                # init annotation
                annotation=self.annotation()
                for j in range(facenum):
                    line = [float(x) for x in self.data.readline().strip().split()]
                    bbox = list(line[:4])
                    if len(line)>4:
                        line[6], line[9], line[12], line[15], line[18] = int(line[6]), int(line[9]), int(line[12]), int(line[15]), int(line[18])
                        index = [line[6], line[9], line[12], line[15], line[18]]
                        self.num_keypoints = len(np.minimum(index, 1))
                        annotation['keypoints'] = line[4:-1]  # 默认为可见 v=2
                        annotation['num_keypoints'] = self.num_keypoints
                    annotation["image_id"] = self.num
                    annotation["id"] = self.annID
                    annotation["bbox"] = bbox
                    annotation['area'] = bbox[2]*bbox[3]
                    annotation['segmentation'] = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
                    self.annotations.append(annotation)

                    self.annID += 1  # 对应对象
                    annotation = self.annotation() # 计算下一个对象

                self.num+=1 # 对应图像
                self.images.append(image)

        jsdata={"info":self.info(),"licenses":self.licenses(),"images":self.images,
                "annotations":self.annotations, "categories":[self.categorie()]}
        json.dump(jsdata,open(self.save_json_path,'w'), indent=4, default=float) # python3 需加上default=float 否则会报错


img_path = 'val'
txt_path = 'val.txt'
save_path = 'keypoints_val.json'

Keypoints2COCO(txt_path, save_path, img_path)()
