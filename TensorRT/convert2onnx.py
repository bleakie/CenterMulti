import argparse
import glob
import json
import logging
import math
import os
import time
from collections import OrderedDict

import cv2
import numpy as np
import onnxruntime as nxrun
import torch

import _init_paths
from config import cfg, update_config
from models.model import create_model
from utils.image import get_affine_transform

logger = logging.getLogger(__name__)

def pre_process(image, cfg=None, scale=1, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width  = int(width * scale)
    mean = np.array(cfg.DATASET.MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(cfg.DATASET.STD, dtype=np.float32).reshape(1, 1, 3)

    inp_height, inp_width = cfg.MODEL.INPUT_H, cfg.MODEL.INPUT_W
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0


    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    images = torch.from_numpy(images)
    meta = {'c': c, 's': s, 
            'out_height': inp_height // cfg.MODEL.DOWN_RATIO, 
            'out_width': inp_width // cfg.MODEL.DOWN_RATIO}
    return images, meta

           
def main(cfg):

    model = create_model('mobilenetv2', cfg.MODEL.HEAD_CONV, cfg).cuda()

    weight_path = '../models/centerface/mobilenetv2-large/model_best.pth'
    state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)['state_dict']
    model.load_state_dict(state_dict)

    onnx_file_path = "../models/centerface/mobilenetv2-large/mobile.onnx"
    
    #img = cv2.imread('test_image.jpg')
    image = cv2.imread('../images/image1.jpeg')
    images, meta = pre_process(image, cfg, scale=1)

    model.cuda()
    model.eval()
    model.float()
    torch_input = images.cuda()
    # print(torch_input.shape)
    print('save...')
    torch.onnx.export(model, torch_input, onnx_file_path, verbose=False)
    sess = nxrun.InferenceSession(onnx_file_path)

    print('save done')
    input_name = sess.get_inputs()[0].name
    output_onnx = sess.run(None, {input_name:  images.cpu().data.numpy()})

  
if __name__ == '__main__':
    config_name = '../experiments/mobilenetv2_512x512.yaml'
    update_config(cfg, config_name)
    main(cfg)
