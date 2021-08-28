import torch
import cv2
import numpy as np
from matplotlib.colors import hsv_to_rgb
from skimage.measure import regionprops
from skimage import measure, morphology
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect


# Use dynamic programming to connect candidate bounding boxes into a smooth box sequence in video
def smooth_bbox_dp(bboxes, length, gap=3):
    bbox_feedback = []

    bbox_index = 0
    bbox_found_num = 0
    bbox_not_random = []
    for frame_index in range(gap, length - gap, gap):

        bboxs = bboxes[bbox_index]
        # Do statistics for the number of frames having valid candidate boxes
        if len(bboxs) > 0:
            bbox_found_num += 1
            # Cache all candidate boxes and their frame index
            bbox_not_random.append((bboxs, frame_index))
        bbox_index += 1

    # Now begin to use dynamic programming to find the optimal path for bbox sequences
    # When implementing, we try to find the minimum distance (identical to finding the maximum reward in paper)

    # Reward for appending new candidate bbox (-K in supplementary material)
    bbox_reward = -0.091
    # Minimum distance from virtual node -1 to node i
    min_distance_dp = [[bbox_reward] * len(bbox_not_random[0][0])]
    # Last bbox index to achieve minimum path distance (namely maximum reward)
    last_bbox_cut = [[(-1, -1)] * len(bbox_not_random[0][0])]
    # Maximum gap for frame connection
    max_dp_gap = 100

    # The main DP loop
    for nr_index in range(1, len(bbox_not_random)):
        bboxs, frame_index = bbox_not_random[nr_index]
        min_distance_dp_this = []
        last_bbox_cut_this = []
        for bbox in bboxs:
            # If directly connect from virtual node -1
            min_distance = bbox_reward
            min_distance_index = (-1, -1)
            for dp_index in range(max(0, nr_index-max_dp_gap), nr_index):
                last_bboxs, last_frame_index = bbox_not_random[dp_index]
                for sub_index in range(len(last_bboxs)):
                    last_bbox = last_bboxs[sub_index]
                    # The modified DIoU as box transition reward
                    mod_diou = DIOU_modify(bbox, last_bbox)
                    # Main Logics:
                    #      1. greater IOU -> larger overlapping -> more likely to be a sequence
                    #      2. smaller distance penalty -> less skip -> more likely to be a sequence
                    iou_reward = -mod_diou
                    # Note that we will give constant reward for adding one more candidate box in the sequence
                    distance = min_distance_dp[dp_index][sub_index] + iou_reward + bbox_reward
                    # Record the selected path
                    if distance <= min_distance:
                        min_distance = distance
                        min_distance_index = (dp_index, sub_index)
            # Record middle results for DP path
            min_distance_dp_this.append(min_distance)
            last_bbox_cut_this.append(min_distance_index)
        # Record middle results for DP path
        min_distance_dp.append(min_distance_dp_this)
        last_bbox_cut.append(last_bbox_cut_this)

    # Now find the last bbox in sequence (the candidate box where the maximum reward path ends)
    last_index = (len(bbox_not_random)-1, 0)
    min_distance = min_distance_dp[last_index[0]][last_index[1]]
    for nr_index in range(len(bbox_not_random) - 1, -1, -1):
        for sub_index in range(len(bbox_not_random[nr_index][0])):
            if min_distance_dp[nr_index][sub_index] <= min_distance:
                last_index = (nr_index, sub_index)
                min_distance = min_distance_dp[nr_index][sub_index]

    # Now track back the selected candidate boxes in the maximum reward path to form a box sequence
    picked_bbox = []
    while last_index[1] != -1:
        bboxs, frame_index = bbox_not_random[last_index[0]]
        picked_bbox.insert(0, (bboxs[last_index[1]], frame_index))
        last_index = last_bbox_cut[last_index[0]][last_index[1]]

    # Now begin to smooth the sequence
    last_already_generated = -1
    # The list for all DP-picked frame index
    picked_frame_index = []

    for bbox_picked_index in range(len(picked_bbox)):

        bbox, frame_index = picked_bbox[bbox_picked_index]
        picked_frame_index.append(frame_index)

        # Now begin to smooth the bbox sequence in a video
        # Case 1 : index from last_gen + 1 to frame_index - 1 (candidate boxes in these frames are not selected by DP)
        for j in range(last_already_generated + 1, frame_index):
            if bbox_picked_index == 0:
                # Starting frames before the first DP-selected candidate box
                if min(list(bbox)) < 75:
                    bbox_perturbed = bbox
                else:
                    # Add very small random perturbation (optional)
                    bbox_perturbation = np.random.uniform(-3, 3, size=4)
                    bbox_perturbed = (bbox[0] + bbox_perturbation[0],
                                      bbox[1] + bbox_perturbation[1],
                                      bbox[2] + bbox_perturbation[2],
                                      bbox[3] + bbox_perturbation[3])
                bbox_feedback.append(bbox_perturbed)
            else:
                # Linear interpolation for generating the remaining boxes
                last_bbox, _ = picked_bbox[bbox_picked_index - 1]

                # It is worth mentioning that we wrote a "bug" here purely by mistake (>_<).
                # The ratio here used for linear interpolation (LP) is created as
                #      "ratio = (j - last_already_generated) / (frame_index - last_already_generated)".
                # However, the originally designed logic that we really desire should be
                #      "ratio = (frame_index - j) / (frame_index - last_already_generated)".
                # Thus, we actually utilize a "reversed" version of LP for moving object discovery and all experiments.
                # However, the pseudo boxes generated "wrongly" still evoke the great tracking performance as reported.
                # We ascribe this phenomenon to the robust design of the overall training framework,
                #      so small "mistakes" made here do not have a destructive effect.

                # We reserve the "mistaken logic" here for re-implementation and parameter-coupling issues
                ratio = (j - last_already_generated) / (frame_index - last_already_generated)
                # Replace with the following line of code to get the originally desired linear interpolation
                # ratio = (frame_index - j) / (frame_index - last_already_generated)

                current_bbox = (last_bbox[0] * ratio + bbox[0] * (1 - ratio),
                                last_bbox[1] * ratio + bbox[1] * (1 - ratio),
                                last_bbox[2] * ratio + bbox[2] * (1 - ratio),
                                last_bbox[3] * ratio + bbox[3] * (1 - ratio))
                bbox_feedback.append(current_bbox)

        # Case 2 : index equals to frame_index (the current frame has a candidate box selected by DP)
        bbox_feedback.append(bbox)
        last_already_generated = frame_index

    # Fill in the last bboxes
    pending_num = length - len(bbox_feedback)
    last_bbox = bbox_feedback[-1]
    # Ending frames after the last DP-selected candidate box
    for i in range(pending_num):
        if min(list(last_bbox)) < 50:
            bbox_perturbed = last_bbox
        else:
            # Add very small random perturbation (optional)
            bbox_perturbation = np.random.uniform(-3, 3, size=4)
            bbox_perturbed = (last_bbox[0] + bbox_perturbation[0],
                              last_bbox[1] + bbox_perturbation[1],
                              last_bbox[2] + bbox_perturbation[2],
                              last_bbox[3] + bbox_perturbation[3])
        bbox_feedback.append(bbox_perturbed)

    assert length == len(bbox_feedback)

    # Now do statistics and calculate various related metrics
    # Average box vary in box sequence (not utilized at last, deprecated)
    total_vary = 0
    for i in range(length - 1):
        current_bbox = bbox_feedback[i]
        next_bbox = bbox_feedback[i + 1]
        for j in range(len(current_bbox)):
            total_vary += abs(current_bbox[j] - next_bbox[j])

    aver_vary = total_vary / (length - 1)

    # Frequency for candidate bboxes be picked by dp (used to calculate video quality score Q_v)
    # According to our experiment, this value is very important for picking high-quality videos in dataset
    bbox_picked_freq = len(picked_bbox) / len(bboxes)

    # Frequency for candidate bboxes are found in a video (not utilized at last, deprecated)
    bbox_found_freq = bbox_found_num / len(bboxes)

    return bbox_feedback, picked_frame_index, bbox_found_freq, bbox_picked_freq, aver_vary


# Calculate IoU score
def IOU(bbox1, bbox2):
    # Bbox: x1, y1, x2, y2

    # Computing area of each rectangles
    S_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    S_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Computing the sum_area
    sum_area = S_bbox1 + S_bbox2

    # Find the each edge of intersect rectangle
    left_line = max(bbox1[1], bbox2[1])
    right_line = min(bbox1[3], bbox2[3])
    top_line = max(bbox1[0], bbox2[0])
    bottom_line = min(bbox1[2], bbox2[2])

    # Judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


# The modified DIoU
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

    # Distance penalty for DIoU
    u = inter_diag / outer_diag
    # IoU
    iou = inter_area / union

    # The original diou
    # diou = iou - u

    # The modified diou, re-weighting distance penalty
    diou = iou - u * 4.1
    if diou < 0:
        diou = diou * 3

    return diou


# Visualize optical flow
def flow_to_image(flow, max_flow=256):
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    return (im * 255).astype(np.uint8)


# Binarize flow map and extract candidate box
def flow_to_bbox(flow, cut_ratio=1/32):

    h, w, c = flow.shape
    # Cut the flows in the margin, as we find that they are always noisy
    flow_clip = flow[int(h * cut_ratio):int(h * (1 - cut_ratio)),
                int(w * cut_ratio):int(w * (1 - cut_ratio))]

    # Calculate average flow within the spatial dimension
    flow_aver = np.mean(flow_clip, axis=(0, 1))
    h_c, w_c, c_c = flow_clip.shape
    flow_aver_tile = np.tile(flow_aver, (h_c, w_c, 1))

    # Calculate L2 distance, for generating the distance map D
    d = np.square(flow_clip - flow_aver_tile)
    distance = np.sqrt(np.sum(d, axis=2))

    # Maximum distance and mean distance in the distance map
    max_distance = distance.max()
    mean_distance = distance.mean()

    max_bboxs = []

    # The weight for centerness, namely \beta in our paper
    center_weights = [0.5, 0.5]
    # The param for calculating the threshold of binarizing distance map, namely \alpha in our paper
    # In practice, we pick two candidate boxes in a frame with two different thresholds
    mean_max_ratios = [0.7, 0.84]
    # The param for judging whether to search for candidate boxes based on the salient motion assumption
    # In practice, we do not collect candidate boxes from those frames without areas of distinguishing motion
    # So there may be some frames having 0 candidate boxes (somewhat different from the simplified algorithm in paper)!
    # However, this mechanism does no harm to the logic, as DP can automatically conduct the optimal box selection
    saliency_param = 2.5
    # Pick top k bboxs in regions (Top 1 by default, so do not worry about this param)
    top_n = 1

    for mean_max_ratio, center_weight in zip(mean_max_ratios, center_weights):
        # Conversion from flow maps to candidate boxes
        bboxs = flow_to_bbox_single_group_param(distance, mean_distance, max_distance,
                                                center_weight=center_weight,
                                                mean_max_ratio=mean_max_ratio, saliency_param=saliency_param,
                                                top_n=top_n)
        max_bboxs.extend(bboxs)

    # Note that the flow map is cropped without image margin, so here we convert the boxes to the actual boxes
    for i in range(len(max_bboxs)):
        max_bbox = max_bboxs[i]
        max_bbox = (max_bbox[0] + cut_ratio * w, max_bbox[1] + cut_ratio * h,
                    max_bbox[2] + cut_ratio * w, max_bbox[3] + cut_ratio * h)
        max_bboxs[i] = max_bbox

    # list of (x1, y1, x2, y2)
    return max_bboxs


# using distance map to generate candidate boxes
def flow_to_bbox_single_group_param(distance, mean_distance, max_distance,
                                    center_weight, mean_max_ratio,
                                    saliency_param, top_n, area_weight=1, small_ratio=0.02, border_ratio=0.7):

    h_c, w_c = distance.shape
    max_dis_index = np.unravel_index(np.argmax(distance), distance.shape)

    max_bboxs = []
    max_scores = []

    # The binarization module includes several tricks, tuning this module is a kind of experiential work
    # The main logic is the same as in the paper, but some trivial but complex operations are omitted

    # In practice, those areas whose motion patterns are not salient enough are not viewed as candidate boxes
    # So there may be some frames having 0 candidate boxes (somewhat different from the simplified algorithm in paper)!
    # However, this mechanism does no harm to the logic, as DP can automatically conduct the optimal box selection
    if mean_distance < 0.05 or max_distance / mean_distance > saliency_param:
        # Notice that here we consider both maximum and mean distance for calculating threshold
        threshold = mean_max_ratio * mean_distance + (1 - mean_max_ratio) * max_distance
        # Generate binary mask
        mask = distance >= threshold
        mask = morphology.remove_small_objects(mask, 80)
        mask = morphology.remove_small_holes(mask, 80)
        labels = measure.label(mask)

        for region in regionprops(labels):
            bbox = region.bbox

            # Skip connected areas with h or w too small
            if (bbox[2] - bbox[0]) < h_c * small_ratio \
                    or (bbox[3] - bbox[1]) < w_c * small_ratio:
                continue
            # Skip very small connect areas
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < 50:
                continue

            # Ultimate goal: find a large bbox, floating at center
            center_score = center_weight * min(h_c - bbox[2], bbox[0]) * min(w_c - bbox[3], bbox[1])
            area_score = area_weight * area
            # Referring to the box score in the paper
            score = center_score + area_score

            # The follows codes include some tricks we find (maybe) helpful through continuous experiments
            # Max distance award (the box with maximum distance pixel inside obtains score *=2)
            if bbox[0] <= max_dis_index[0] <= bbox[2] and bbox[1] <= max_dis_index[1] <= bbox[3]:
                score *= 2
            # Corner penalty (the bbox with at least one edge at the corner are more likely to be a noisy box)
            if min(h_c - bbox[2], bbox[0]) <= 15:
                score /= 2
                # drop long bboxes lying at border
                if (bbox[3] - bbox[1]) > w_c * border_ratio:
                    continue
            if min(w_c - bbox[3], bbox[1]) <= 15:
                score /= 2
                # drop long bboxes lying at border
                if (bbox[2] - bbox[0]) > h_c * border_ratio:
                    continue

            # Drop extreme rectangles
            if bbox[2] == bbox[0] or (bbox[3] - bbox[1]) / (bbox[2] - bbox[0]) > 6:
                continue

            if bbox[3] == bbox[1] or (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]) > 6:
                continue

            # Find top k bboxes with highest score, while by default top_n = 1
            found_bbox_num = len(max_bboxs)
            insert_index = found_bbox_num
            for index in range(found_bbox_num - 1, -1, -1):
                if score > max_scores[index]:
                    insert_index = index
                else:
                    break

            # Insert bbox and score
            if insert_index < top_n:
                max_bboxs.insert(insert_index, (bbox[1], bbox[0], bbox[3], bbox[2]))
                max_scores.insert(insert_index, score)
            # Drop exceeded bboxes
            if len(max_bboxs) > top_n:
                max_bboxs = max_bboxs[:top_n]
                max_scores = max_scores[:top_n]

    return max_bboxs


# Calculate frame quality score, important for filtering boxes of high quality
def calc_nearby_bbox_freq(picked_frame_index, video_length, search_range=None, gap=3):

    # We have tested to use 3 and 10 for calculating frame quality score
    # According to the experiment, we find that short interval (3) is better than long interval (10) in most cases
    if search_range is None or len(search_range) == 0:
        search_range = [3, 10]
    # Remember that Flow+DP is conducted on a sub-sampled video with a sub-sampling ratio of 'gap'
    search_range = [s * gap for s in search_range]

    # Init for doing statistics
    freq_dicts = [[0] * video_length for _ in range(len(search_range))]
    freq_collect_max = [[0] * video_length for _ in range(len(search_range))]

    # Do statistic for the number of adjacent frames (of a certain frame) potential to be selected by DP
    for r_i in range(len(search_range)):
        for v_i in range(gap, video_length - gap, gap):
            left_index = max(0, v_i - search_range[r_i])
            right_index = min(video_length - 1, v_i + search_range[r_i])
            for sub_i in range(left_index, right_index + 1):
                # increment count
                current = freq_collect_max[r_i][sub_i]
                freq_collect_max[r_i][sub_i] = current + 1

    # Do statistic for the number of adjacent frames (of a certain frame) indeed selected by DP
    for r_i in range(len(search_range)):
        for v_i in picked_frame_index:
            left_index = max(0, v_i - search_range[r_i])
            right_index = min(video_length - 1, v_i + search_range[r_i])
            for sub_i in range(left_index, right_index + 1):
                # increment count
                current = freq_dicts[r_i][sub_i]
                freq_dicts[r_i][sub_i] = current + 1

    # Calculate the frequency of DP selection within all adjacent frames (of a certain frame)
    feedback = []
    for v_i in range(video_length):
        score = []
        for r_i in range(len(search_range)):
            if freq_collect_max[r_i][v_i] != 0:
                score.append(float(freq_dicts[r_i][v_i] / freq_collect_max[r_i][v_i]))
            else:
                score.append(float(0.0))
        feedback.append(score)
    return feedback


# Calculate the percentage of boxes lying at corner
# As an implementation detail, we do not need corner boxes since they tend to be awful
def calc_corner_bbox_freq(smoothed_bboxs, img_shape, cut_ratio=1/32):

    # Do statistics for corner bboxs
    corner_bbox_num = 0
    # Extreme axis, [x_left, y_top, x_right, y_bottom]
    axis_extreme = [int(cut_ratio * img_shape[1]), int(cut_ratio * img_shape[0]),
                    int((1-cut_ratio) * img_shape[1]), int((1-cut_ratio) * img_shape[0])]

    for bbox in smoothed_bboxs:
        x1, y1, x2, y2 = bbox
        x_at_corner = (x1 < axis_extreme[0] + 10) or (x2 > axis_extreme[2] - 10)
        y_at_corner = (y1 < axis_extreme[1] + 10) or (y2 > axis_extreme[3] - 10)
        # The bboxes at four corners, with weight of 1
        if x_at_corner and y_at_corner:
            corner_bbox_num += 1
        # The bboxes stick to the margin, with weight of 0.3
        elif x_at_corner or y_at_corner:
            corner_bbox_num += 0.3

    return corner_bbox_num / len(smoothed_bboxs)


def resize_flow(flow, new_shape):
    _, _, h, w = flow.shape
    new_h, new_w = new_shape
    flow = torch.nn.functional.interpolate(flow, (new_h, new_w),
                                           mode='bilinear', align_corners=True)
    scale_h, scale_w = h / float(new_h), w / float(new_w)
    flow[:, 0] /= scale_w
    flow[:, 1] /= scale_h
    return flow


def flow_warp(x, flow12, pad='border', mode='bilinear'):
    B, _, H, W = x.size()

    base_grid = mesh_grid(B, H, W).type_as(x)  # B2HW

    v_grid = norm_grid(base_grid + flow12)  # BHW2
    if 'align_corners' in inspect.getfullargspec(torch.nn.functional.grid_sample).args:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad, align_corners=True)
    else:
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
    return im1_recons


def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid


def norm_grid(v_grid):
    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def restore_model(model, pretrained_file):
    epoch, weights = load_checkpoint(pretrained_file)

    model_keys = set(model.state_dict().keys())
    weight_keys = set(weights.keys())

    # load weights by name
    weights_not_in_model = sorted(list(weight_keys - model_keys))
    model_not_in_weights = sorted(list(model_keys - weight_keys))
    if len(model_not_in_weights):
        print('Warning: There are weights in model but not in pre-trained.')
        for key in (model_not_in_weights):
            print(key)
            weights[key] = model.state_dict()[key]
    if len(weights_not_in_model):
        print('Warning: There are pre-trained weights not in model.')
        for key in (weights_not_in_model):
            print(key)
        from collections import OrderedDict
        new_weights = OrderedDict()
        for key in model_keys:
            new_weights[key] = weights[key]
        weights = new_weights

    model.load_state_dict(weights)
    return model


def load_checkpoint(model_path):
    weights = torch.load(model_path)
    epoch = None
    if 'epoch' in weights:
        epoch = weights.pop('epoch')
    if 'state_dict' in weights:
        state_dict = (weights['state_dict'])
    else:
        state_dict = weights
    return epoch, state_dict
