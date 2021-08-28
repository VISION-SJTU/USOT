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
parser.add_argument('--dir', type=str, default='/data/dataset/lasot/', help='your lasot data dir')
parser.add_argument('--output', type=str, default='lasot_{}.json', help='the parsed lasot annotation file')
parser.add_argument('--bbox', type=str, default='flow', help='the methodology of generating bbox')
args = parser.parse_args()

sub_sets = sorted({'train', 'val'})

def parse():

    lasot_benchmark_path = join(args.dir, "LaSOTBenchmark")
    lasot_testing_set_path = join(args.dir, "testing_set.txt")
    sub_sets = sorted(listdir(lasot_benchmark_path))

    testing_set_file = open(lasot_testing_set_path, "r", encoding='utf-8')
    testing_videos = testing_set_file.readlines()
    testing_videos = [video.strip('\n') for video in testing_videos]

    ts = inference.init_module()
    # In practice, Flow+DP is conducted on sub-sampled videos in LaSOT, here gap is the sub-sampling rate
    gap = 3
    lasot = []

    for si, sub_set in enumerate(sub_sets):

        sub_set_base_path = join(lasot_benchmark_path, sub_set)
        videos = sorted(listdir(sub_set_base_path))
        s = []
        for vi, video in enumerate(videos):

            # WARNING: never use videos in lasot testing split !!!!
            if video in testing_videos:
                continue

            print('subset: {}, video {}, id: {:04d} / {:04d}'.format(sub_set, video,
                        si * len(videos) + vi, len(videos) * len(sub_sets)))

            video_base_path = join(sub_set_base_path, video)

            # Get image size
            im_path = join(video_base_path, 'img', '00000001.jpg')
            im = cv2.imread(im_path)
            size = im.shape  # height, width
            frame_sz = [size[1], size[0]]  # width, height

            # Get all image names
            all_frames = sorted(glob.glob(join(video_base_path, 'img', '*.jpg')))

            # Since lasot videos are too long, we have to split them into pieces
            piece_length = 200
            extend_length = 20

            # Begin to window-scan the whole video
            split_id = 0
            while True:

                # Calculate the video piece
                start_frame = split_id * piece_length
                end_frame = start_frame + piece_length + extend_length
                if start_frame >= len(all_frames):
                    break

                split_id += 1

                # Border case: the last video piece
                if end_frame >= len(all_frames):
                    end_frame = len(all_frames) - 1
                    start_frame = end_frame - piece_length - extend_length
                frames_picked = all_frames[start_frame:end_frame+1]

                v = dict()
                v['base_path'] = join(sub_set, video)
                v['frame'] = []

                start = time.time()
                try:
                    # The main logic for sampling pseudo boxes with DP+Flow
                    bboxs, picked_frame_index, stat_tuple = \
                        inference.inference_sequence(ts, frames_picked, gap=gap, vis=False)
                    freq_dict, bbox_found_freq, bbox_picked_freq, aver_vary, corner_bbox_freq = stat_tuple
                except:
                    print('=====WARN====== subset: {} video id: {:04d} gen bbox fails.'.format(sub_set, vi))
                    continue

                print("Split {}, start frame {}, end frame {}.".format(split_id, start_frame, end_frame))
                print("bbox_found_freq: {}, bbox_picked_freq: {}, vary_aver: {}, corner_bbox_freq: {}, consumed time: {} seconds.".format(
                    bbox_found_freq, bbox_picked_freq, aver_vary, corner_bbox_freq, time.time()-start))
                v["aver_vary"] = aver_vary
                v["bbox_found_freq"] = bbox_found_freq
                v["bbox_picked_freq"] = bbox_picked_freq
                v['picked_frame_index'] = picked_frame_index
                v['corner_bbox_freq'] = corner_bbox_freq

                # Very bad bbox sequences, which can never be used
                if bbox_picked_freq < 0.35 or corner_bbox_freq > 0.4 \
                        or (bbox_picked_freq - 1/3 * corner_bbox_freq) < 0.33:
                    continue

                for i in range(len(frames_picked)):
                    f = dict()
                    f['frame_sz'] = frame_sz
                    f['img_path'] = frames_picked[i].split('/')[-1]
                    o = dict()
                    o['trackid'] = split_id - 1
                    o['bbox'] = [int(bboxs[i][0]), int(bboxs[i][1]), int(bboxs[i][2]), int(bboxs[i][3])]
                    # The 2-element list of [short-term DP-pick-freq, long-term DP-pick-freq],
                    #            respectively estimated with T_s as [3, 10]
                    o['confidence'] = freq_dict[i]
                    f['objs'] = [o]
                    v['frame'].append(f)

                s.append(v)

        lasot.append(s)

    print('save json (raw lasot info), please wait 1 min~')
    json.dump(lasot, open(args.output.format(args.bbox), 'w'), indent=4, sort_keys=True)
    print('{}.json has been saved in ./'.format(args.output))


if __name__ == "__main__":
    parse()
