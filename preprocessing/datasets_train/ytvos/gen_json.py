from os.path import join
import json
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str,
                    default='/home/jlzheng/dataset/tracking/usot/ytvos_{}', help='your ytvos anno file output dir')
parser.add_argument('--anno_file', type=str, default='ytvos_{}.json', help='the parsed ytvos annotation file')
parser.add_argument('--bbox', type=str, default='flow', help='the methodology of generating bbox')
args = parser.parse_args()

print('loading json (raw ytvos info), please wait 20 seconds~')
ytvos = json.load(open(args.anno_file.format(args.bbox), 'r'))
output_dir = args.output_dir.format(args.bbox)

def gen_json():

    snippets = dict()
    n_snippets = 0
    n_videos = 0
    for subset in ytvos:
        for video in ytvos[subset]:
            n_videos += 1
            frames = video['frame']
            # Set of object id in video
            id_set = []
            # Corresponding frames of objects, at most 60 objects
            id_frames = [[]] * 60
            for f, frame in enumerate(frames):
                objs = frame['objs']
                # frame_sz = frame['frame_sz']
                for obj in objs:
                    trackid = obj['trackid']
                    # bbox = obj['bbox']
                    if trackid not in id_set:
                        id_set.append(trackid)
                        id_frames[trackid] = []
                    id_frames[trackid].append(f)
            if len(id_set) > 0:
                snippets[video['base_path']] = dict()
            for selected in id_set:
                frame_ids = sorted(id_frames[selected])
                sequences = np.split(frame_ids, np.array(np.where(np.diff(frame_ids) > 1)[0]) + 1)
                sequences = [s for s in sequences if len(s) > 1]  # remove isolated frame.
                for seq in sequences:
                    snippet = dict()
                    for frame_id in seq:
                        frame = frames[frame_id]
                        for obj in frame['objs']:
                            if obj['trackid'] == selected:
                                o = obj
                                continue
                        if 'confidence' not in o:
                            snippet[frame['img_path'].split('.')[0]] = o['bbox']
                        else:
                            content = o['bbox'] + o['confidence']
                            snippet[frame['img_path'].split('.')[0]] = content
                    trackid_str = '{:02d}'.format(selected)
                    snippets[video['base_path']][trackid_str] = snippet
                    n_snippets += 1
                    meta_data = {
                        'aver_vary': video['aver_vary'],
                        'base_path': video['base_path'],
                        'bbox_found_freq': video['bbox_found_freq'],
                        'bbox_picked_freq': video['bbox_picked_freq'],
                        'corner_bbox_freq': video['corner_bbox_freq'],
                        'frame_sz': frames[0]['frame_sz']
                    }
                    snippets[video['base_path']][trackid_str]["meta"] = meta_data
            if n_videos % 100 == 0:
                print('video: {:d} snippets_num: {:d}'.format(n_videos, n_snippets))

    # If we use optical flow to generate bbox, we have to do some another pre-works for data_loader
    if args.bbox == 'flow':

        search_gap = 1
        max_frame_gap = 220
        print_gap = 50
        should_drop_videos = []

        video_idx = 0
        video_names = list(snippets.keys())
        for video_name in snippets:

            # Print the process
            if video_idx % print_gap == 0:
                print("Preprocessing video {}/{}".format(video_idx, len(video_names)))
            video_idx += 1

            all_obj_seqs = snippets[video_name]
            valid_obj_num = 0
            for track_id in all_obj_seqs:
                bbox_seq = all_obj_seqs[track_id]
                bbox_picked_freq = bbox_seq['meta']['bbox_picked_freq']
                corner_bbox_freq = bbox_seq['meta']['corner_bbox_freq']
                frame_sz = bbox_seq['meta']['frame_sz']
                # Drop extreme bbox sequences which are unreliable
                if bbox_picked_freq < 0.35 or \
                    corner_bbox_freq > 0.4 or \
                        bbox_picked_freq - 1/3 * corner_bbox_freq < 0.33:
                    continue
                else:
                    valid_obj_num += 1
                frame_ids = list(bbox_seq.keys())
                if 'meta' in frame_ids:
                    frame_ids.remove('meta')

                for idx in range(len(frame_ids)):

                    if idx > 0:
                        last_right_ptr = bbox_seq[frame_ids[idx-1]][7]

                        # If the current frame has already been covered by previous search
                        if last_right_ptr >= idx:
                            corner_score = calc_corner_score(bbox_seq[frame_ids[idx]][:4], frame_sz)
                            last_left_ptr = bbox_seq[frame_ids[idx-1]][6]
                            snippets[video_name][track_id][frame_ids[idx]] += [last_left_ptr,
                                                                               last_right_ptr, corner_score]
                            continue

                    # Parameters for searching lower and upper bound
                    # These parameters are somewhat different from those of other datasets,
                    #         as YT-VOS naturally has a natural frame sub-sampling rate of 5.
                    iou_threshold = 0.40
                    quality_threshold = 0.35

                    # Begin to search from idx to obtain T_l (lower bound for memory frames)
                    left_ptr = idx - search_gap
                    prev_bbox_info = bbox_seq[frame_ids[idx]]
                    while True:
                        if left_ptr < max(0, idx - max_frame_gap):
                            left_ptr += search_gap
                            break
                        current_bbox_info = bbox_seq[frame_ids[left_ptr]]
                        mod_DIOU = DIOU_modify(current_bbox_info[:4], prev_bbox_info[:4])

                        # Drop if a sequence is not smooth, or the bbox is not reliable
                        if mod_DIOU < iou_threshold or current_bbox_info[4] <= quality_threshold:
                            left_ptr += search_gap
                            break
                        left_ptr -= search_gap
                        prev_bbox_info = current_bbox_info

                    # Begin to search from idx to obtain T_u (upper bound for memory frames)
                    right_ptr = idx + search_gap
                    prev_bbox_info = bbox_seq[frame_ids[idx]]
                    while True:
                        if right_ptr >= min(len(frame_ids), idx + max_frame_gap):
                            right_ptr -= search_gap
                            break
                        current_bbox_info = bbox_seq[frame_ids[right_ptr]]
                        mod_DIOU = DIOU_modify(current_bbox_info[:4], prev_bbox_info[:4])

                        # Drop if a sequence is not smooth, or the bbox is not reliable
                        if mod_DIOU < iou_threshold or current_bbox_info[4] <= quality_threshold:
                            right_ptr -= search_gap
                            break
                        right_ptr += search_gap
                        prev_bbox_info = current_bbox_info

                    left_ptr = min(left_ptr + search_gap // 2, idx)
                    right_ptr = max(right_ptr - search_gap // 2, idx)
                    corner_score = calc_corner_score(bbox_seq[frame_ids[idx]][:4], frame_sz)
                    snippets[video_name][track_id][frame_ids[idx]] += [left_ptr, right_ptr, corner_score]

            # If no valid sequences remaining
            if valid_obj_num == 0:
                should_drop_videos.append(video_name)

        for video_name in should_drop_videos:
            del snippets[video_name]

    train = {k: v for (k, v) in snippets.items() if 'train' in k}
    json.dump(train, open(join(output_dir, 'train.json'), 'w'), indent=4, sort_keys=True)
    print('done!')


def DIOU_modify(bbox1, bbox2):
    bbox1 = np.array(list(bbox1))
    bbox2 = np.array(list(bbox2))

    w1 = bbox1[2] - bbox1[0]
    h1 = bbox1[3] - bbox1[1]
    w2 = bbox2[2] - bbox2[0]
    h2 = bbox2[3] - bbox2[1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bbox1[2] + bbox1[0]) / 2
    center_y1 = (bbox1[3] + bbox1[1]) / 2
    center_x2 = (bbox2[2] + bbox2[0]) / 2
    center_y2 = (bbox2[3] + bbox2[1]) / 2

    inter_max_xy = np.min(np.stack([bbox1[2:], bbox2[2:]]), axis=0)
    inter_min_xy = np.max(np.stack([bbox1[:2], bbox2[:2]]), axis=0)
    out_max_xy = np.max(np.stack([bbox1[2:], bbox2[2:]]), axis=0)
    out_min_xy = np.min(np.stack([bbox1[:2], bbox2[:2]]), axis=0)

    inter = np.clip((inter_max_xy - inter_min_xy), a_min=0, a_max=5000)
    inter_area = inter[0] * inter[1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = np.clip((out_max_xy - out_min_xy), a_min=0, a_max=5000)
    outer_diag = (outer[0] ** 2) + (outer[1] ** 2)
    union = area1 + area2 - inter_area

    u = inter_diag / outer_diag
    iou = inter_area / union

    # The original diou
    # diou = iou - u

    # The modified diou
    diou = iou - u * 2.5

    return diou


# We give priority to boxes lying in the middle of the image
def calc_corner_score(bbox, frame_sz, cut_ratio=1/32):
    # Extreme axis, [x_left, y_top, x_right, y_bottom]
    axis_extreme = [int(cut_ratio * frame_sz[0]), int(cut_ratio * frame_sz[1]),
                    int((1-cut_ratio) * frame_sz[0]), int((1-cut_ratio) * frame_sz[1])]

    x_border = min(abs(bbox[0] - axis_extreme[0]), abs(axis_extreme[2] - bbox[2]))
    y_border = min(abs(bbox[1] - axis_extreme[1]), abs(axis_extreme[3] - bbox[3]))
    corner_score = min(x_border / (axis_extreme[2] - axis_extreme[0]), 0.1) + \
                   min(y_border / (axis_extreme[3] - axis_extreme[1]), 0.1)
    return corner_score


if __name__ == "__main__":
    gen_json()
