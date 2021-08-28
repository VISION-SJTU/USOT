from os.path import join, isdir, exists
from os import listdir, mkdir, makedirs
import cv2
import numpy as np
import glob
from concurrent import futures
import sys
import time
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='/data/dataset/got10k', help='your got10k data dir')
parser.add_argument('--anno_file', type=str, default='got10k_{}.json', help='the parsed got10k annotation file')
parser.add_argument('--output_dir', type=str,
                    default='/home/jlzheng/dataset/tracking/usot/got10k_{}', help='your got10k output dir')
parser.add_argument('--bbox', type=str, default='flow', help='the methodology of generating bbox')
parser.add_argument('--instance_size', type=int, default=511, help='your instance size')
parser.add_argument('--num_threads', type=int, default=24, help='your number of threads')
args = parser.parse_args()

data_base_path = args.input_dir
sub_sets = ['train', 'val']

# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    format_str = "{0:." + str(decimals) + "f}"
    percents = format_str.format(100 * (iteration / float(total)))
    filled_length = int(round(barLength * iteration / float(total)))
    bar = '' * filled_length + '-' * (barLength - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]   # width, height
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x


def crop_video(sub_set, video, crop_path, instanc_size):
    video_crop_base_path = join(crop_path, video['base_path'])
    if not isdir(video_crop_base_path):
        os.makedirs(video_crop_base_path)

    video_file_base = join(data_base_path, video['base_path'])

    for frame in video["frame"]:
        frame_file_path = join(video_file_base, frame["img_path"])

        im = cv2.imread(frame_file_path)
        avg_channels = np.mean(im, axis=(0, 1))

        # Use generated bbox in parse_got10k.py
        filename = frame["img_path"].split(".")[0]
        for object in frame["objs"]:
            bbox = object["bbox"]
            track_id = object["trackid"]

            z, x = crop_like_SiamFC(im, bbox, instanc_size=instanc_size, padding=avg_channels)
            cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(int(filename), track_id)), z)
            cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(filename), track_id)), x)


def main(instanc_size=511, num_threads=24):

    crop_path = '{}/crop{:d}'.format(args.output_dir.format(args.bbox), instanc_size)
    anno_file = args.anno_file.format(args.bbox)
    anno = json.load(open(anno_file, 'r'))

    if not isdir(crop_path):
        os.makedirs(crop_path)

    for i in range(len(sub_sets)):
        sub_set = sub_sets[i]
        anno_sub = anno[i]
        n_videos = len(anno_sub)
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_video, sub_set, video, crop_path, instanc_size) for video in anno_sub]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_videos, prefix=sub_set, suffix='Done ', barLength=40)


if __name__ == '__main__':
    since = time.time()
    main(args.instance_size, args.num_threads)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

