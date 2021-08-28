from os.path import join
from os import listdir
import json, time
import glob
import argparse
import xml.etree.ElementTree as ET
import numpy as np
from preprocessing.flow_module import inference

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='/data/dataset/ILSVRC_VID', help='your vid data dir')
parser.add_argument('--output', type=str, default='vid_{}.json', help='the parsed vid annotation file')
parser.add_argument('--bbox', type=str, default='flow', help='the methodology of generating bbox')
args = parser.parse_args()


def parse():
    VID_base_path = args.dir
    ann_base_path = join(VID_base_path, 'Annotations/VID/train/')

    sub_sets = sorted({'a', 'b', 'c', 'd', 'e'})

    ts = inference.init_module()

    # In practice, Flow+DP is conducted on sub-sampled videos in VID, here gap is the sub-sampling rate
    gap = 3

    vid = {}
    for sub_set in sub_sets:
        sub_set_base_path = join(ann_base_path, sub_set)
        videos = sorted(listdir(sub_set_base_path))
        s = []
        for vi, video in enumerate(videos):

            print('subset: {} video id: {:04d} / {:04d}'.format(sub_set, vi, len(videos)))

            v = dict()
            v['base_path'] = join(sub_set, video)
            v['frame'] = []
            video_base_path = join(sub_set_base_path, video)
            xmls = sorted(glob.glob(join(video_base_path, '*.xml')))

            all_frames = []
            f_cache = []
            count = 0
            for xml in xmls:
                # At most pick 2000 frames, for limited GPU memory
                if count >= 2000:
                    break
                f = dict()
                xmltree = ET.parse(xml)
                size = xmltree.findall('size')[0]
                frame_sz = [int(it.text) for it in size]  # width, height
                f['frame_sz'] = frame_sz
                f['img_path'] = xml.split('/')[-1].replace('xml', 'JPEG')
                img_full_path = xml.replace("Annotations", "Data").replace(".xml", ".JPEG")
                all_frames.append(img_full_path)
                f_cache.append(f)
                count += 1

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

            for i in range(len(f_cache)):
                f = f_cache[i]
                o = dict()
                o['trackid'] = 0
                o['bbox'] = [int(bboxs[i][0]), int(bboxs[i][1]), int(bboxs[i][2]), int(bboxs[i][3])]
                # The 2-element list of [short-term DP-pick-freq, long-term DP-pick-freq],
                #            respectively estimated with T_s as [3, 10]
                o['confidence'] = freq_dict[i]
                f['objs'] = [o]
                v['frame'].append(f)

            s.append(v)

        vid[sub_set] = s

    print('save json (raw vid info), please wait 1 min~')
    json.dump(vid, open(args.output.format(args.bbox), 'w'), indent=4, sort_keys=True)
    print('{}.json has been saved in ./'.format(args.output))


if __name__ == "__main__":
    parse()
