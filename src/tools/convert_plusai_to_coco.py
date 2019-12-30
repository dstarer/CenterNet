from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

import pickle
import json
import numpy as np
import cv2
from PIL import Image

DATA_PATH = '/media/andy/andy-2TB/2d_labeling/2d_detection_data/'
DEBUG = False

'''
# Values Name Description
1 type  'Car', 'Bus', 'Truck', 'Pedestrian', 'Bicycle', 'Traffic Cone', 'barricade', 'moto', 'other' 

'''


def _center_to_coco_bbox(bbox):
    """
    from center format
    to (tlx, tly, w, h)
    :param bbox:
    :return:
    """
    tlx = bbox[0] - bbox[2] / 2
    tly = bbox[1] - bbox[3] / 2
    return [(tlx), (tly), (bbox[2]), (bbox[3])]


def _corner_to_coco_bbox(bbox):
    """
    from corner format
    to (tlx, tly, w, h)
    :param bbox:
    :return:
    """
    return [(bbox[0]), (bbox[1]), (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]


def load_json_data(fpath):
    with open(fpath, "r") as f:
        json_data = json.load(f)
    return json_data


def dump_json(json_data, fpath):
    with open(fpath, "w") as f:
        json.dump(json_data, f)


cats = ['car', 'bus', 'truck', 'moto', 'pedestrian', 'bike', 'traffic_cone', 'barricade', 'other']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
cat_info = []
for i, cat in enumerate(cats):
    cat_info.append({'name': cat, 'id': i + 1})

ret = {"images": [], "annotations": [], "categories": cat_info}

labeling_path = DATA_PATH + "train/train.json"
# labeling_path = DATA_PATH + "test/test.json"
json_data = load_json_data(labeling_path)

image_to_id = {}
max_image_id = 0

for item in json_data:
    img_abs_path = DATA_PATH + "images/" + item['file']
    try:
        image = cv2.imread(img_abs_path)
        image_height, image_width, _ = image.shape
    except Exception as e:
        print(e)
        print("skip image, path ", img_abs_path)
        continue
    print(item['file'])
    image_id = image_to_id.get(item['file'], -1)
    if image_id == -1:
        image_id = max_image_id
        image_to_id.setdefault(item['file'], image_id)
        max_image_id += 1

    image_info = {'file_name': item['file'],
                  'id': image_id}

    ret['images'].append(image_info)

    for box_id, box in enumerate(item['boxes']):
        cat_id = cat_ids[box['class']]
        truncated = 1 if 'truncated_vehicle' in box and box['truncated_vehicle'] == 1 else 0
        if 'occluded' not in box or box['occluded'] is None:
            occluded = 0
        elif 'occluded' in box and box['occluded'] is 'partially':
            occluded = 1
        elif 'occluded' in box and box['occluded'] is 'mostly' or box['occluded'] is 'barely':
            occluded = 2
        bbox = [float(box['coord'][0]) * image_width, float(box['coord'][1]) * image_height,
                float(box['coord'][2]) * image_width, float(box['coord'][3]) * image_height]

        ann = {
            'image_id': image_id,
            'id': int(len(ret['annotations']) + 1),
            'category_id': cat_id,
            'bbox': _center_to_coco_bbox(bbox),
            'truncated': truncated,
            'occluded': occluded
        }
        ret['annotations'].append(ann)

print("# images: ", len(ret['images']))
print("# annotations: ", len(ret['annotations']))

out_path = "{}/annotations/plusai_{}.json".format(DATA_PATH, "train")
dump_json(ret, out_path)

