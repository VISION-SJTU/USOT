# -*- coding:utf-8 -*-
# ! ./usr/bin/env python

import cv2
import json
import glob
import time
import numpy as np
from os.path import join
from os import listdir

import argparse
from preprocessing.flow_module import inference

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='/data/dataset/got10k', help='your got10k data dir')
parser.add_argument('--output', type=str, default='got10k_{}.json', help='the parsed got10k annotation file')
parser.add_argument('--bbox', type=str, default='flow', help='the methodology of generating bbox')
args = parser.parse_args()

got10k_base_path = args.dir
sub_sets = sorted({'train', 'val'})

def parse():

    ts = inference.init_module()
    # In practice, Flow+DP is conducted on sub-sampled videos in GOT-10K, here gap is the sub-sampling rate
    gap = 3
    got10k = []

    for sub_set in sub_sets:
        sub_set_base_path = join(got10k_base_path, sub_set)
        videos = sorted(listdir(sub_set_base_path))
        s = []
        for vi, video in enumerate(videos):
            # Abandon files such as list.txt
            if ".txt" in video:
                continue
            print('subset: {} video id: {:04d} / {:04d}'.format(sub_set, vi, len(videos)))
            v = dict()
            v['base_path'] = join(sub_set, video)
            v['frame'] = []
            video_base_path = join(sub_set_base_path, video)

            # Get image size
            im_path = join(video_base_path, '00000001.jpg')
            im = cv2.imread(im_path)
            size = im.shape  # height, width
            frame_sz = [size[1], size[0]]  # width, height

            # Get all image names
            all_frames = sorted(glob.glob(join(video_base_path, '*.jpg')))
            if len(all_frames) > 2000:
                all_frames = all_frames[:2000]

            start = time.time()
            try:
                # The main logic for sampling pseudo boxes with DP+Flow
                bboxs, picked_frame_index, stat_tuple = \
                    inference.inference_sequence(ts, all_frames, gap=gap, vis=False)
                freq_dict, bbox_found_freq, bbox_picked_freq, aver_vary, corner_bbox_freq = stat_tuple
            except:
                print('=====WARN====== subset: {} video id: {:04d} gen bbox fails.'.format(sub_set, vi))
                continue
            print("bbox_found_freq: {}, bbox_picked_freq: {}, vary_aver: {}, corner_bbox_freq: {}, consumed time: {} seconds.".format(
                    bbox_found_freq, bbox_picked_freq, aver_vary, corner_bbox_freq, time.time() - start))
            v["aver_vary"] = aver_vary
            v["bbox_found_freq"] = bbox_found_freq
            v["bbox_picked_freq"] = bbox_picked_freq
            v['picked_frame_index'] = picked_frame_index
            v['corner_bbox_freq'] = corner_bbox_freq

            for i in range(len(all_frames)):
                f = dict()
                f['frame_sz'] = frame_sz
                f['img_path'] = all_frames[i].split('/')[-1]
                o = dict()
                o['trackid'] = 0
                o['bbox'] = [int(bboxs[i][0]), int(bboxs[i][1]), int(bboxs[i][2]), int(bboxs[i][3])]
                # The 2-element list of [short-term DP-pick-freq, long-term DP-pick-freq],
                #        estimated with T_s as [3, 10]
                o['confidence'] = freq_dict[i]
                f['objs'] = [o]
                v['frame'].append(f)

            s.append(v)

        got10k.append(s)

    print('save json (raw got10k info), please wait 1 min~')
    json.dump(got10k, open(args.output.format(args.bbox), 'w'), indent=4, sort_keys=True)
    print('{}.json has been saved in ./'.format(args.output))


if __name__ == "__main__":
    parse()
