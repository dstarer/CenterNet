from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import sys

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import os
import cv2
import json
from lib.opts import opts
from lib.detectors.detector_factory import detector_factory
from lib.datasets.dataset_factory import get_dataset
import time
import fnmatch
image_ext = ['jpg', 'jpeg', 'png', 'webp']


def load_benchmark_file(opt):
    fp_benchmark = opt.data_dir + "/benchmark/" + opt.benchmark
    print("load benchmark file ", fp_benchmark)
    image_names = []
    if os.path.isfile(fp_benchmark):
        with open(fp_benchmark) as fp:
            json_data = json.load(fp)
            for item in json_data:
                image_names.append(item['file'])
    return image_names


def recursive_glob(rootdir='.', pattern='*'):
    """
    search recursively for files matching a specified pattern.
    """
    rootdir = os.path.join(rootdir, '')
    matches = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            p = os.path.join(root, filename)
            matches.append(p[len(rootdir): ])
    
    return matches

def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 0)
    
    print(opt.dataset)
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)

    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    # image_names = load_benchmark_file(opt)
    rootdir = opt.data_dir + "/images/" + opt.benchmark
    print("load benchmark dir ", rootdir)
    image_names = recursive_glob(rootdir)
    p_output = opt.data_dir + "/prediction/ctdet_" + opt.benchmark + ".csv"
    # p_output = opt.data_dir + "/prediction" + opt.output
    with open(p_output, "w") as f:
        # f.write("img,id,class,tlx,tly,brx,bry,score\n")
        for (image_name) in image_names:
            image_path = os.path.join(rootdir, image_name)
            try:
                img = cv2.imread(image_path)
                image_height, image_width, _ = img.shape
            except Exception as e:
                print("file read error, path ", image_path)
                continue
            print("try to detect on ", image_path)
            ret = detector.run(image_path)
            detection = ret['results']
            for key, items in detection.items():
                for i in range(items.shape[0]):
                    cx = (items[i, 0] + items[i, 2]) / 2.0
                    cy = (items[i, 1] + items[i, 3]) / 2.0
                    w = items[i, 2] - items[i, 0]
                    h = items[i, 3] - items[i, 1]
                    cx /= image_width
                    w /= image_width
                    cy /= image_height
                    h /= image_height
                    line = image_name + "," + str(cx) + "," + str(cy) + "," + str(w) + "," + str(h) + "," + str(
                        items[i, 4]) + "," + str(key - 1) + "\n"
                    f.write(line)
            if len(detection) == 0:
                f.write(image_name + "\n")


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
