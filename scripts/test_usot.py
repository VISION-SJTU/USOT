# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Revised by Jilai Zheng, for USOT
# ------------------------------------------------------------------------------
from __future__ import absolute_import
import os
import cv2
import random
import argparse
import numpy as np
import sys
from os.path import exists, join, dirname, realpath, abspath

import lib.models.models as models

from lib.tracker.usot_tracker import USOTTracker
from easydict import EasyDict as edict
from lib.utils.train_utils import load_pretrain
from lib.utils.test_utils import cxy_wh_2_rect, get_axis_aligned_bbox, poly_iou
from lib.dataset_loader.benchmark import load_dataset


def parse_args():
    """
    args for USOT testing.
    """
    parser = argparse.ArgumentParser(description='USOT testing')
    parser.add_argument('--arch', dest='arch', default='USOT', help='backbone architecture')
    parser.add_argument('--resume', default='var/snapshot/USOT_star.pth', type=str, help='pretrained model')
    parser.add_argument('--dataset', default='VOT2018', help='dataset test')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    parser.add_argument('--version', default='v1', help='testing style version')
    args = parser.parse_args()

    return args


def track(tracker, net, video, args):
    start_frame, toc = 0, 0

    # Save result to evaluate
    if args.epoch_test:
        suffix = args.resume.split('/')[-1]
        suffix = suffix.split('.')[0]
        tracker_path = os.path.join('var/result', args.dataset, args.arch + suffix)
    else:
        tracker_path = os.path.join('var/result', args.dataset, args.arch)

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in args.dataset:
        baseline_path = os.path.join(tracker_path, 'baseline')
        video_path = os.path.join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    elif 'GOT' in args.dataset:
        video_path = os.path.join(tracker_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
        time_path = os.path.join(video_path, '{}_time.txt'.format(video['name']))
    else:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

    if os.path.exists(result_path):
        return

    regions = []
    track_times = []
    image_files, gt = video['image_files'], video['gt']
    for f, image_file in enumerate(image_files):

        im = cv2.imread(image_file)
        if len(im.shape) == 2:
            # Align with training
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        tic = cv2.getTickCount()

        # Init procedure
        if f == start_frame:
            cx, cy, w, h = get_axis_aligned_bbox(gt[f])
            target_pos = np.array([cx, cy])
            target_sz = np.array([w, h])
            # Init tracker
            state = tracker.init(im, target_pos, target_sz, net)
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(1 if 'VOT' in args.dataset else gt[f])

        # Tracking procedure
        elif f > start_frame:
            state = tracker.track(state, im)
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            b_overlap = poly_iou(gt[f], location) if 'VOT' in args.dataset else 1
            if b_overlap > 0:
                regions.append(location)
            else:
                regions.append(2)
                start_frame = f + 5
        else:
            regions.append(0)

        toc += cv2.getTickCount() - tic
        if 'GOT' in args.dataset:
            track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())

    with open(result_path, "w") as fin:
        if 'VOT' in args.dataset:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')
        else:
            for x in regions:
                p_bbox = x.copy()
                fin.write(
                    ','.join([str(i + 1) if idx == 0 or idx == 1 else str(i) for idx, i in enumerate(p_bbox)]) + '\n')

    if 'GOT' in args.dataset:
        with open(time_path, 'w') as file_handle:
            for x in track_times:
                file_handle.write("{:.6f}\n".format(x))

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps'.format(video['name'], toc, f / toc))


def main():
    args = parse_args()

    # Prepare model
    net = models.__dict__[args.arch]()
    net = load_pretrain(net, args.resume)
    net.eval()
    net = net.cuda()

    # Prepare video
    dataset = load_dataset(args.dataset)
    video_keys = list(dataset.keys()).copy()

    # Prepare tracker
    info = edict()
    info.arch = args.arch
    info.dataset = args.dataset
    info.epoch_test = args.epoch_test
    info.version = args.version

    if info.arch == 'USOT':
        tracker = USOTTracker(info)
    else:
        assert False, "Warning: Model should be USOT, but currently {}.".format(info.arch)

    # Tracking all videos in benchmark
    for video in video_keys:
        track(tracker, net, dataset[video], args)


if __name__ == '__main__':
    main()
