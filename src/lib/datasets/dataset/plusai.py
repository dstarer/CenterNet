from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
# sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

import torch.utils.data as data
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import torch
import json
import cv2
import os
import math

cats = ['__background__', 'car', 'bus', 'truck', 'moto', 'pedestrian', 'bike', 'traffic_cone', 'barricade', 'other']


class PlusAI(data.Dataset):
    num_classes = 9
    default_resolution = [512, 512]
    mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(PlusAI, self).__init__()
        self.data_dir = opt.data_dir
        print("---data dir ", self.data_dir)
        self.img_dir = os.path.join(self.data_dir, "images")
        # training & validation.
        if split == 'val':
            self.annot_path = os.path.join(self.data_dir, "annotations", "plusai_val.json")
        elif split == 'train':
            self.annot_path = os.path.join(
                self.data_dir, "annotations", "plusai_train.json")
        else:
            self.annot_path = os.path.join(self.data_dir, "annotations", 'plusai_benchmark.json').format(split)

        self.max_objs = 128

        self.class_name = cats
        self._valid_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self.split = split
        self.opt = opt

        print("===> initializing plusai {} data.".format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        # self.num_samples = min(15000, self.num_samples)
        print("Loaded {} {} samples".format(split, self.num_samples))

    def __len__(self):
        return self.num_samples

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_boxes):
        detections = []
        for image_id in all_boxes:
            for cls_ind in all_boxes[image_id]:
                category_id = cls_ind
                for bbox in all_boxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))
                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

