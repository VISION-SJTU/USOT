from os.path import join
from os import listdir
import json, time
import glob
import argparse
import numpy as np
from preprocessing.flow_module import inference
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='/data/dataset/youtube_vos', help='your youtube-vos data dir')
parser.add_argument('--output', type=str, default='ytvos_{}.json', help='your youtube-vos data dir')
parser.add_argument('--bbox', type=str, default='flow', help='the methodology of generating bbox')
args = parser.parse_args()


def parse():
    ytvos_base_path = args.dir

    sub_sets = sorted({'train'})

    ts = inference.init_module()

    # In practice, Flow+DP is conducted on sub-sampled videos in YT-VOS
    # Since YT-VOS is originally provided with a frame interval of 5,
    #         the flow estimating gap is chosen to be 1 other than 3 in other datasets
    gap = 1
    # The initial frame interval for generating optical flow (dynamically changing according to the specific video)
    init_flow_interval = 1

    ytvos = {}
    for sub_set in sub_sets:
        sub_set_base_path = join(ytvos_base_path, sub_set, 'JPEGImages')
        videos = sorted(listdir(sub_set_base_path))
        s = []
        for vi, video in enumerate(videos):

            print('subset: {} video id: {:04d} / {:04d}'.format(sub_set, vi, len(videos)))

            v = dict()
            v['base_path'] = join(sub_set, video)
            v['frame'] = []
            video_base_path = join(sub_set_base_path, video)

            # Get all image names
            all_frames = sorted(glob.glob(join(video_base_path, '*.jpg')))
            # Get image size
            im_path = all_frames[0]
            im = cv2.imread(im_path)
            size = im.shape  # height, width
            frame_sz = [size[1], size[0]]  # width, height
            if len(all_frames) > 2000:
                all_frames = all_frames[:2000]

            start = time.time()
            try:
                # The main logic for sampling pseudo boxes with DP+Flow
                bboxs, picked_frame_index, stat_tuple = \
                    inference.inference_sequence(ts, all_frames,
                                                 gap=gap, init_adjacent=init_flow_interval, vis=False)
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
                #            respectively estimated with T_s as [3, 10]
                o['confidence'] = freq_dict[i]
                f['objs'] = [o]
                v['frame'].append(f)

            s.append(v)

        ytvos[sub_set] = s

    print('save json (raw yt-vos info), please wait ~')
    json.dump(ytvos, open(args.output.format(args.bbox), 'w'), indent=4, sort_keys=True)
    print('{}.json has been saved in ./'.format(args.output))


if __name__ == "__main__":
    parse()
