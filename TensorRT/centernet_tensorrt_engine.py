import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

import _init_paths
from config import cfg, update_config
from models.utils import _gather_feat, _transpose_and_gather_feat
from tensorrt_model import TRTModel
from utils.image import get_affine_transform, transform_preds


class CenterNetTensorRTEngine(object):
    def __init__(self, config_file, weight_file):

        update_config(cfg, config_file)
        self.cfg = cfg        
        self.trtmodel = TRTModel(weight_file)

    def preprocess(self, image, scale=1, meta=None):
        height, width = image.shape[0:2]
        new_height = int(height * scale)
        new_width = int(width * scale)
        mean = np.array(self.cfg.DATASET.MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.cfg.DATASET.STD, dtype=np.float32).reshape(1, 1, 3)

        inp_height, inp_width = self.cfg.MODEL.INPUT_H, self.cfg.MODEL.INPUT_W
        c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (new_width, new_height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)

        meta = {'c': c, 's': s,
                'out_height': inp_height // self.cfg.MODEL.DOWN_RATIO,
                'out_width': inp_width // self.cfg.MODEL.DOWN_RATIO}

        return np.ascontiguousarray(images), meta

    def run(self, imgs):

        images , meta = self.preprocess(imgs)
        hm, wh, hps, reg, hm_hp, hp_offset = self.trtmodel(images)
 
        predictions = self.postprocess(hm, wh, hps, reg, hm_hp, hp_offset, meta)
        
        return predictions

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _topk(self, scores, K=40):
        batch, cat, height, width = scores.size()
          
        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys   = (topk_inds / width).int().float()
        topk_xs   = (topk_inds % width).int().float()
          
        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / K).int()
        topk_inds = _gather_feat(
            topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
 
    def _topk_channel(self, scores, K=40):
          batch, cat, height, width = scores.size()
          
          topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

          topk_inds = topk_inds % (height * width)
          topk_ys   = (topk_inds / width).int().float()
          topk_xs   = (topk_inds % width).int().float()

          return topk_scores, topk_inds, topk_ys, topk_xs
                    
    def multi_pose_decode(self,
        heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
        batch, cat, height, width = heat.size()
        num_joints = kps.shape[1] // 2
        # perform nms on heatmaps
        heat = self._nms(heat)
        scores, inds, clses, ys, xs = self._topk(heat, K=K)

        kps = _transpose_and_gather_feat(kps, inds)
        kps = kps.view(batch, K, num_joints * 2)
        kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
        kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
        if reg is not None:
            reg = _transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        clses  = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)

        bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                          ys - wh[..., 1:2] / 2,
                          xs + wh[..., 0:1] / 2, 
                          ys + wh[..., 1:2] / 2], dim=2)
        if hm_hp is not None:
            hm_hp = self._nms(hm_hp)
            thresh = 0.1
            kps = kps.view(batch, K, num_joints, 2).permute(
              0, 2, 1, 3).contiguous() # b x J x K x 2
            reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
            hm_score, hm_inds, hm_ys, hm_xs = self._topk_channel(hm_hp, K=K) # b x J x K
            if hp_offset is not None:
                hp_offset = _transpose_and_gather_feat(
                  hp_offset, hm_inds.view(batch, -1))
                hp_offset = hp_offset.view(batch, num_joints, K, 2)
                hm_xs = hm_xs + hp_offset[:, :, :, 0]
                hm_ys = hm_ys + hp_offset[:, :, :, 1]
            else:
                hm_xs = hm_xs + 0.5
                hm_ys = hm_ys + 0.5

            mask = (hm_score > thresh).float()
            hm_score = (1 - mask) * -1 + mask * hm_score
            hm_ys = (1 - mask) * (-10000) + mask * hm_ys
            hm_xs = (1 - mask) * (-10000) + mask * hm_xs
            hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
              2).expand(batch, num_joints, K, K, 2)
            dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
            min_dist, min_ind = dist.min(dim=3) # b x J x K
            hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
            min_dist = min_dist.unsqueeze(-1)
            min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
              batch, num_joints, K, 1, 2)
            hm_kps = hm_kps.gather(3, min_ind)
            hm_kps = hm_kps.view(batch, num_joints, K, 2)
            l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
            mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
                 (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
                 (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
            mask = (mask > 0).float().expand(batch, num_joints, K, 2)
            kps = (1 - mask) * hm_kps + mask * kps
            kps = kps.permute(0, 2, 1, 3).contiguous().view(
              batch, K, num_joints * 2)
        detections = torch.cat([bboxes, scores, kps, torch.transpose(hm_score.squeeze(dim=3), 1, 2)], dim=2)
        return detections
        
    def multi_pose_post_process(self, dets, c, s, h, w):
        # dets: batch x max_dets x 40
        # return list of 39 in image coord
        ret = []
        for i in range(dets.shape[0]):
            bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
            pts = transform_preds(dets[i, :, 5:15].reshape(-1, 2), c[i], s[i], (w, h))
            top_preds = np.concatenate(
                [bbox.reshape(-1, 4), dets[i, :, 4:5], 
                pts.reshape(-1, 10), dets[i, :, 15:20]], axis=1).astype(np.float32).tolist()
            ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
        return ret
            
    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets = self.multi_pose_post_process(
          dets.copy(), [meta['c']], [meta['s']],
          meta['out_height'], meta['out_width'])
        for j in range(1, self.cfg.MODEL.NUM_CLASSES + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 20)
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:] /= scale
        return dets[0]
                
    def postprocess(self, hm, wh, hps, reg, hm_hp, hp_offset, meta):

        hm = hm.sigmoid_()
        hm_hp = hm_hp.sigmoid_()
        detections = self.multi_pose_decode(hm, wh, hps, reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.cfg.TEST.TOPK)
        dets = self.post_process(detections, meta, 1)
        return dets
