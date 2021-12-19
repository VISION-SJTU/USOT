import os
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from lib.utils.track_utils import load_yaml, get_subwindow_tracking, python2round, im_to_torch
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class USOTTracker(object):
    def __init__(self, info):
        super(USOTTracker, self).__init__()
        # The model and benchmark info
        self.info = info

        self.aug_frame_init = iaa.Sequential([
            iaa.Fliplr(1),
        ])

    def init(self, im, target_pos, target_sz, model):

        # Use PrPool to collect template feature
        model.pr_pool = True

        state = dict()
        # Epoch test
        p = USOTConfig()

        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]

        # Find the config file for test
        # Hyper-parameters for all datasets are kept the same throughout our experiments.
        absPath = os.path.abspath(os.path.dirname(__file__))
        yname = '{}.yaml'.format(self.info.arch)
        yamlPath = os.path.join(absPath, '../../experiments/test/{0}'.format(yname))
        cfg_benchmark = load_yaml(yamlPath, subset=True)
        # Update hyper-parameters according to config file
        p.update(cfg_benchmark)
        p.renew()

        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = cfg_benchmark['big_sz']
            p.renew()
        else:
            p.instance_size = cfg_benchmark['small_sz']
            p.renew()

        # Search feature size designed to be the same as response map size in USOT v1)
        p.sf_size = p.score_size
        # Generate grids
        self.grids(p)

        net = model

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = round(np.sqrt(wc_z * hc_z))

        # Pool the template deep feature with PrPool from the first frame
        avg_chans = np.mean(im, axis=(0, 1))
        z_crop, crop_info = get_subwindow_tracking(im, target_pos, p.exemplar_size,
                                                   s_z, avg_chans, target_sz, need_bbox=True, vis=False)
        bbox = crop_info['template_bbox']
        template_bbox = self.pool_label_template(p, bbox)
        template_bbox = torch.tensor([template_bbox]).float().cuda()

        z = z_crop.unsqueeze(0)
        net.template(z.cuda(), template_bbox=template_bbox)

        # The cosine window
        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))  # [17,17]
        elif p.windowing == 'uniform':
            window = np.ones(int(p.score_size), int(p.score_size))

        state['p'] = p
        state['net'] = net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz

        # Online memory-based module
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        # Begin to initialize the memory queue
        # Add pooled features of init frame in the memory queue
        x_crop, crop_info = get_subwindow_tracking(im, target_pos, p.instance_size,
                                                   python2round(s_x), avg_chans, target_sz,
                                                   out_mode='raw', need_bbox=True, vis=False)
        search_bbox = crop_info['template_bbox']
        x_crop_unsq = im_to_torch(x_crop.copy())
        x_crop_unsq = x_crop_unsq.unsqueeze(0)
        search_bbox_pool = self.pool_label_search(p, search_bbox)
        search_bbox_pool = torch.tensor([search_bbox_pool]).float().cuda()
        memory_feature = net.extract_memory_feature(ori_x=x_crop_unsq.cuda(), search_bbox=search_bbox_pool)
        memory_feature = memory_feature.cpu().detach()

        # Add pooled features of left/right flipped init template in the first frame in the memory queue
        search_bbox_aug = BoundingBoxesOnImage([
            BoundingBox(x1=search_bbox[0], y1=search_bbox[1], x2=search_bbox[2], y2=search_bbox[3]),
        ], shape=x_crop.shape)
        x_crop_aug, search_bbox_aug = self.aug_frame_init(image=x_crop, bounding_boxes=search_bbox_aug)
        search_bbox_aug = [self.clip_number(search_bbox_aug[0].x1, _max=x_crop_aug.shape[0]),
                           self.clip_number(search_bbox_aug[0].y1, _max=x_crop_aug.shape[1]),
                           self.clip_number(search_bbox_aug[0].x2, _max=x_crop_aug.shape[0]),
                           self.clip_number(search_bbox_aug[0].y2, _max=x_crop_aug.shape[1])]
        x_crop_aug = im_to_torch(x_crop_aug.copy())
        x_crop_aug_unsq = x_crop_aug.unsqueeze(0)
        search_bbox_aug_pool = self.pool_label_search(p, search_bbox_aug)
        search_bbox_aug_pool = torch.tensor([search_bbox_aug_pool]).float().cuda()
        memory_feature_aug = net.extract_memory_feature(ori_x=x_crop_aug_unsq.cuda(),
                                                        search_bbox=search_bbox_aug_pool)
        memory_feature_aug = memory_feature_aug.cpu().detach()

        # Init frame template patch and its left/right flip will absolutely be picked in memory queue
        state['init_features'] = [memory_feature, memory_feature_aug]
        # Deep feature for the init frame template patches
        state['memory_features'] = [memory_feature]
        state['memory_confidences'] = [0.9]

        return state

    def update(self, net, x_crops, target_pos, target_sz, window,
               scale_z, p, template_mem=None, score_mem=None):

        # Track with the model
        cls_score, bbox_pred, cls_memory, xf = net.track(x_crops, template_mem=template_mem, score_mem=score_mem)
        cls_score = F.sigmoid(cls_score).squeeze().cpu().data.numpy()
        cls_memory = F.sigmoid(cls_memory).squeeze().cpu().data.numpy()
        # Aggregate online cls module and offline cls module
        cls_score = p.ratio * cls_score + (1 - p.ratio) * cls_memory

        # The bbox predicted
        bbox_pred = bbox_pred.squeeze().cpu().data.numpy()

        pred_x1 = self.grid_to_search_x - bbox_pred[0, ...]
        pred_y1 = self.grid_to_search_y - bbox_pred[1, ...]
        pred_x2 = self.grid_to_search_x + bbox_pred[2, ...]
        pred_y2 = self.grid_to_search_y + bbox_pred[3, ...]

        # Add size penalty and scale penalty
        s_c = self.change(self.sz(pred_x2 - pred_x1, pred_y2 - pred_y1) / (self.sz_wh(target_sz)))
        # Add ratio penalty
        r_c = self.change((target_sz[0] / target_sz[1]) / ((pred_x2 - pred_x1) / (pred_y2 - pred_y1)))

        penalty = np.exp(-(r_c * s_c - 1) * p.penalty_k)
        pscore = penalty * cls_score

        # Add window penalty
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence

        # Get max score index
        r_max, c_max = np.unravel_index(pscore.argmax(), pscore.shape)

        # Conversion of bbox to real size
        pred_x1 = pred_x1[r_max, c_max]
        pred_y1 = pred_y1[r_max, c_max]
        pred_x2 = pred_x2[r_max, c_max]
        pred_y2 = pred_y2[r_max, c_max]

        pred_xs = (pred_x1 + pred_x2) / 2
        pred_ys = (pred_y1 + pred_y2) / 2
        pred_w = pred_x2 - pred_x1
        pred_h = pred_y2 - pred_y1

        diff_xs = pred_xs - p.instance_size // 2
        diff_ys = pred_ys - p.instance_size // 2

        diff_xs, diff_ys, pred_w, pred_h = diff_xs / scale_z, diff_ys / scale_z, pred_w / scale_z, pred_h / scale_z

        target_sz = target_sz / scale_z

        # The size learning rate
        lr = penalty[r_max, c_max] * cls_score[r_max, c_max] * p.lr

        # The size rate
        res_xs = target_pos[0] + diff_xs
        res_ys = target_pos[1] + diff_ys
        res_w = pred_w * lr + (1 - lr) * target_sz[0]
        res_h = pred_h * lr + (1 - lr) * target_sz[1]

        target_pos = np.array([res_xs, res_ys])
        target_sz = target_sz * (1 - lr) + lr * np.array([res_w, res_h])

        # extract memory feature of the current frame with predicted bbox
        pred_bbox_to_prpool = self.pool_label_search(p, [pred_x1, pred_y1, pred_x2, pred_y2])
        pred_bbox_to_prpool = torch.tensor([pred_bbox_to_prpool]).float().cuda()
        feature_mem = net.extract_memory_feature(xf=xf, search_bbox=pred_bbox_to_prpool)
        feature_mem = feature_mem.cpu().detach()
        return target_pos, target_sz, cls_score[r_max, c_max], feature_mem

    def track(self, state, im):
        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        # Crop the search frame
        x_crop, _ = get_subwindow_tracking(im, target_pos, p.instance_size, python2round(s_x), avg_chans)
        x_crop = x_crop.unsqueeze(0)

        memory_features = state['memory_features']
        memory_confidences = state['memory_confidences']
        template_mem = state['init_features'].copy()
        score_mem = [0.9, 0.9]
        mem_length = len(memory_confidences)
        # Only (N_q-3) memorized features are collected from intermediate results
        mem_queue_size_update = p.mem_queue_size - 3

        if mem_length <= 1:
            template_mem += ([memory_features[0]] * (mem_queue_size_update + 1))
            score_mem += ([memory_confidences[0]] * (mem_queue_size_update + 1))
            template_mem = torch.cat(template_mem, dim=0)
            score_mem = torch.tensor(score_mem).unsqueeze(0)
        else:
            # Online sample (N_q - 3) memory templates with the highest confidence scores
            gap = (mem_length - 1) / mem_queue_size_update
            for i in range(mem_queue_size_update):
                # 2021.12.19: We notice that the calculation of start_index and end_index seems to deviate
                #             from what we expect. We leave the implementation version here for reproducing issues.
                start_index = min(int(int(i * gap) * mem_length), mem_length - 1)
                end_index = min(int(int((i + 1) * gap) * mem_length), mem_length - 1)
                if start_index >= end_index:
                    template_mem.append(memory_features[start_index])
                    score_mem.append(memory_confidences[start_index])
                else:
                    score_tmp = np.array(memory_confidences[start_index:end_index])
                    # Pick (N_q - 3) frames with highest confidence scores
                    max_index = np.argmax(score_tmp) + start_index
                    template_mem.append(memory_features[max_index])
                    score_mem.append(memory_confidences[max_index])
            # Always pick the last memory template
            template_mem.append(memory_features[-1])
            score_mem.append(memory_confidences[-1])
            template_mem = torch.cat(template_mem, dim=0)
            score_mem = torch.tensor(score_mem).unsqueeze(0)

        target_pos, target_sz, confidence, feat_mem = self.update(net, x_crop.cuda(), target_pos,
                                                                  target_sz * scale_z, window, scale_z, p,
                                                                  template_mem=template_mem.cuda(),
                                                                  score_mem=score_mem.cuda())

        # Update the memory queue
        state['memory_features'].append(feat_mem)
        state['memory_confidences'].append(confidence)

        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['cls_score'] = confidence
        state['p'] = p

        return state

    def clip_number(self, num, _max=127.0, _min=0.0):

        if num >= _max:
            return _max
        elif num <= _min:
            return _min
        else:
            return num

    def grids(self, p):
        """
        Each element of feature map on template patch and response map
        :return: H*W*2 (position for each element)
        """
        sz = p.score_size
        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search_x = x * p.total_stride + p.instance_size // 2
        self.grid_to_search_y = y * p.total_stride + p.instance_size // 2

        # Template feature map grid
        tf_sz = p.tf_size
        sz_x_t = tf_sz // 2
        sz_y_t = tf_sz // 2
        x, y = np.meshgrid(np.arange(0, tf_sz) - np.floor(float(sz_x_t)),
                           np.arange(0, tf_sz) - np.floor(float(sz_y_t)))

        self.grid_to_template = {}
        self.grid_to_template_x = x * p.total_stride + p.exemplar_size // 2
        self.grid_to_template_y = y * p.total_stride + p.exemplar_size // 2

        # Axis for search area feature map
        sf_sz = p.sf_size
        sf_sz_x = sf_sz // 2
        self.search_area_x_axis = (np.arange(0, sf_sz) - np.floor(float(sf_sz_x))) \
                                  * p.total_stride + p.instance_size // 2

    def pool_label_template(self, p, bbox):

        reg_min = self.grid_to_template_x[0][0]
        reg_max = self.grid_to_template_x[-1][-1]
        bbox = np.array(bbox, np.float32)
        bbox = np.clip(bbox, a_max=reg_max, a_min=reg_min)
        sz = 2 * (p.tf_size // 2)
        slope = sz / (reg_max - reg_min)
        return (bbox - reg_min) * slope

    def pool_label_search(self, p, bbox):
        """
        Convert bbox from the image axis to the search area axis

        In the original design of USOT, response map and search area share the same axis for PrPool boxes (25*25).
        Such design is consistent during both training and testing, and produces the performance as reported.

        However, this design may cause a feature misalignment problem,
                  as the actual sizes of response map and search area are different (25*25 and 31*31 respectively).

        Here we simply reserve the original design in both training and testing codes,
                  making sure you can obtain the same results on the benchmarks as reported in the paper.
        """

        reg_min = self.search_area_x_axis[0]
        reg_max = self.search_area_x_axis[-1]
        sz = 2 * (p.sf_size // 2)
        slope = sz / (reg_max - reg_min)
        gap = 1.0 / slope
        bbox = np.array(bbox, np.float32)
        bbox = np.clip(bbox, a_max=reg_max + gap, a_min=reg_min - gap)
        return (bbox - reg_min) * slope

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(self, wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)


class USOTConfig(object):
    penalty_k = 0.021
    window_influence = 0.321
    lr = 0.730
    windowing = 'cosine'
    exemplar_size = 127
    instance_size = 255
    total_stride = 8
    score_size = (instance_size - exemplar_size) // total_stride + 1 + 8
    context_amount = 0.5

    # Feature size of template patch
    tf_size = 15
    # Feature axis of search area (designed to be the same as response map size in USOT v1)
    sf_size = 25

    # The weight for offline module, indicated by (1-w) in the paper
    ratio = 0.3
    # The memorized feature number online, indicated by N_q in the paper
    mem_queue_size = 7

    def update(self, newparam=None):
        if newparam:
            for key, value in newparam.items():
                setattr(self, key, value)
            self.renew()

    def renew(self):
        self.score_size = (self.instance_size - self.exemplar_size) // self.total_stride + 1 + 8
