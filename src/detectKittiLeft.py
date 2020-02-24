from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import os
import cv2

from lib.opts import opts
from lib.detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']

#output_file = "/media/andy/jinwen-2TB/oxford/2014-06-24-14-15-17/mono_right_2d_detection.csv"
output_file = "/home/andy/Downloads/20191121T154824_j7-wuzhui_auto_1/front_left_camera_detection.csv"


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 0)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if os.path.isdir(opt.demo):
        image_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(opt.demo, file_name))
    else:
        image_names = [opt.demo]
    with open(output_file, "w") as f:
        f.write("img,id,class,tlx,tly,brx,bry,score\n")
        for (image_name) in image_names:
            print(image_name)
            ret = detector.run(image_name)
            detection = ret['results']
            idx = 0
            for key, items in detection.items():
                for i in range(items.shape[0]):
                    line = image_name + "," + str(idx) + "," + str(key) + "," + str(items[i, 0]) + "," + str(items[i, 1]) + "," + str(items[i, 2]) + "," + str(items[i, 3]) + "," + str(items[i, 4]) + "\n"
                    f.write(line)
                    idx += 1


if __name__ == '__main__':
    opt = opts().init()
    main(opt)
