# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Revised by Jilai Zheng, for USOT
# ------------------------------------------------------------------------------
from __future__ import division

import cv2
import json
import random
from os.path import join
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from lib.utils.image_utils import *
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

sample_random = random.Random()


# sample_random.seed(123456)

class USOTDataset(Dataset):
    def __init__(self, cfg):
        super(USOTDataset, self).__init__()
        # Pair information
        self.template_size = cfg.USOT.TRAIN.TEMPLATE_SIZE
        self.search_size = cfg.USOT.TRAIN.SEARCH_SIZE

        # Response map size
        self.size = 25
        # Feature size of template patch
        self.tf_size = 15
        # Feature axis of search area (designed to be the same as response map size in USOT v1)
        self.sf_size = 25
        # Total stride of backbone
        self.stride = cfg.USOT.TRAIN.STRIDE

        # Aug information
        # Aug for template patch
        self.shift = cfg.USOT.DATASET.SHIFT
        self.scale = cfg.USOT.DATASET.SCALE

        # Aug for search areas
        self.shift_s = cfg.USOT.DATASET.SHIFTs
        self.scale_s = cfg.USOT.DATASET.SCALEs

        # Aug for memory search areas
        self.shift_m = cfg.USOT.DATASET.SHIFTm
        self.scale_m = cfg.USOT.DATASET.SCALEm

        # Threshold for video quality
        self.video_quality = cfg.USOT.DATASET.VIDEO_QUALITY
        # Number of memory frames in a single training instance
        self.memory_num = cfg.USOT.TRAIN.MEMORY_NUM
        # Parameter for sampling memory frames
        self.far_sample = cfg.USOT.DATASET.FAR_SAMPLE

        # Choices for training
        # Set self.cycle_memory = False for naive Siamese training
        # Set self.cycle_memory = True for cycle memory training
        self.cycle_memory = True
        # For testing dataloader, you can set self.loader_test to True
        # See the dataloader testing scripts at the bottom of this file
        self.loader_test = False

        self.grids()

        # Augmentation for template patch
        self.template_aug_seq = iaa.Sequential([
            iaa.Fliplr(0.4),
            iaa.Flipud(0.2),
            iaa.PerspectiveTransform(scale=(0.01, 0.07)),
            iaa.CoarseDropout((0.0, 0.05), size_percent=0.15, per_channel=0.5),
            iaa.SaltAndPepper(0.05, per_channel=True),
        ])

        # Augmentation for search area
        self.search_aug_seq = iaa.Sequential([
            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
            iaa.MultiplyBrightness((0.5, 1.5)),
            iaa.MotionBlur(k=(3, 9), angle=[-60, 60]),
        ])

        # Augmentation for memory search areas
        self.memory_aug_seq = iaa.Sequential([
            iaa.Fliplr(0.4),
            iaa.Flipud(0.2),
            iaa.PerspectiveTransform(scale=(0.01, 0.15)),
            iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
            iaa.MultiplyBrightness((0.5, 1.5)),
            iaa.MotionBlur(k=(3, 9), angle=[-60, 60]),
        ])

        # Training data information
        print('train datas: {}'.format(cfg.USOT.TRAIN.WHICH_USE))
        # List of all training datasets
        self.train_datas = []
        start = 0
        self.num = 0
        for data_name in cfg.USOT.TRAIN.WHICH_USE:
            dataset = subData(cfg, data_name, start, self.memory_num,
                              self.video_quality, self.far_sample)
            self.train_datas.append(dataset)
            # Real video number
            start += dataset.num
            # The number used for subset shuffling
            self.num += dataset.num_use

        self._shuffle()
        print(cfg)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        """
        The main logic for sampling training instances.
        Two sampling modes are provided:
           1. naive Siamese: sampling a template frame and crop both template and search area from it
           2. cycle memory: besides the naive Siamese pair, sampling N_mem memory search areas additionally
        Switch between two modes: self.cycle_memory should be set to True only when conducting cycle memory training.
        """
        index = self.pick[index]
        dataset, index = self._choose_dataset(index)

        # For offline naive Siamese tracker (template patch and search area are picked in the same frame)
        # Note that the actual video index may be re-sampled in _get_instances() if the video is of low quality
        # Warning: cycle_memory should be set to True only when conducting cycle memory training.
        pair_info = dataset._get_instances(index, cycle_memory=self.cycle_memory)

        # Here only one template frame image is returned, and it will be utilized for both template and search
        search_image = cv2.imread(pair_info[0])
        search_bbox = self._toBBox(search_image, pair_info[1])
        template_image = search_image
        template_bbox = search_bbox

        # Augmentation for template and search area
        template_aug, bbox_t, dag_param_t = self._augmentation(template_image,
                                                               template_bbox, self.template_size)
        search_aug, bbox_s, dag_param_s = self._augmentation(search_image,
                                                             search_bbox, self.search_size, search=True)

        # Tmp name is used only for local loader testing (see bottom of this file)
        loader_test_name = "{:06d}".format(random.randint(0, 999999))
        # Draw for loader testing
        if self.loader_test:
            self._draw(search_aug, bbox_s, "../../var/loader/" + loader_test_name + "_s.jpg")
            self._draw(template_aug, bbox_t, "../../var/loader/" + loader_test_name + "_t.jpg")

        # Now begin to retrieve memory search areas for cycle memory training
        search_memory = []
        if self.cycle_memory:
            # Get memory frames and its pseudo bounding boxes generated by DP + flow
            search_images_nearby = [cv2.imread(image_path) for image_path in pair_info[2]]
            search_bbox_nearby = [self._toBBox(search_images_nearby[i], pair_info[3][i])
                                  for i in range(len(search_images_nearby))]
            for i in range(len(search_images_nearby)):
                crop_nearby, bbox_nearby, _ = self._augmentation(search_images_nearby[i], search_bbox_nearby[i],
                                                                 self.search_size, search=True,
                                                                 cycle_memory=self.cycle_memory)
                # Draw for loader testing
                if self.loader_test:
                    self._draw(crop_nearby, bbox_nearby, "../../var/loader/" +
                               loader_test_name + "_n_{:02d}.jpg".format(i))
                crop_nearby = np.array(crop_nearby)
                crop_nearby = np.transpose(crop_nearby, (2, 0, 1)).astype(np.float32)
                search_memory.append(crop_nearby)
            search_memory = np.stack(search_memory)

        # From PIL image to numpy
        template = np.array(template_aug)
        search = np.array(search_aug)

        template, search = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search])

        # Pseudo classification labels for offline naive tracker
        out_label = self._dynamic_label([self.size, self.size], dag_param_s.shift)

        # Pseudo regression labels for offline naive tracker
        reg_label, reg_weight = self.reg_label(bbox_s)

        # Template pseudo bbox label for PrROIPooling
        bbox_t = self.pool_label_template(bbox_t)
        bbox_t = np.array(bbox_t, np.float32)

        if len(search_memory) > 0:
            # Search area bbox label for PrROIPooling
            bbox_s = self.pool_label_search(bbox_s)
            bbox_s = np.array(bbox_s, np.float32)
            # Additionally return memory frames and bbox_s for Siamese search areas
            return template, search, out_label, reg_label, reg_weight, bbox_t, search_memory, bbox_s

        return template, search, out_label, reg_label, reg_weight, bbox_t

    def _shuffle(self):
        """
        Random shuffle
        """
        pick = []
        m = 0
        while m < self.num:
            p = []
            for subset in self.train_datas:
                sub_p = subset.pick
                p += sub_p
            sample_random.shuffle(p)

            pick += p
            m = len(pick)
        self.pick = pick
        print("dataset length {}".format(self.num))

    def _choose_dataset(self, index):
        for dataset in self.train_datas:
            if dataset.start + dataset.num > index:
                return dataset, index - dataset.start

    def grids(self):
        """
        Each element of feature map on template patch and response map
        :return: H*W*2 (position for each element)
        """
        # Response map grid
        sz = self.size
        sz_x = sz // 2
        sz_y = sz // 2
        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))
        self.grid_to_search = {}
        self.grid_to_search_x = x * self.stride + self.search_size // 2
        self.grid_to_search_y = y * self.stride + self.search_size // 2

        # Template feature map grid
        tf_sz = self.tf_size
        sz_x_t = tf_sz // 2
        sz_y_t = tf_sz // 2
        x, y = np.meshgrid(np.arange(0, tf_sz) - np.floor(float(sz_x_t)),
                           np.arange(0, tf_sz) - np.floor(float(sz_y_t)))

        self.grid_to_template = {}
        self.grid_to_template_x = x * self.stride + self.template_size // 2
        self.grid_to_template_y = y * self.stride + self.template_size // 2

        # Search area feature map grid
        sf_sz = self.sf_size
        sf_x_s = sf_sz // 2
        self.search_area_x_axis = (np.arange(0, sf_sz) - np.floor(float(sf_x_s))) \
                                  * self.stride + self.search_size // 2

    def reg_label(self, bbox):
        """
        Generate regression label
        :param bbox: [x1, y1, x2, y2]
        :return: [l, t, r, b]
        """
        x1, y1, x2, y2 = bbox
        l = self.grid_to_search_x - x1  # [17, 17]
        t = self.grid_to_search_y - y1
        r = x2 - self.grid_to_search_x
        b = y2 - self.grid_to_search_y

        l, t, r, b = map(lambda x: np.expand_dims(x, axis=-1), [l, t, r, b])
        reg_label = np.concatenate((l, t, r, b), axis=-1)
        reg_label_min = np.min(reg_label, axis=-1)
        inds_nonzero = (reg_label_min > 0).astype(float)

        return reg_label, inds_nonzero

    def pool_label_template(self, bbox):
        """
        Get pseudo bbox for PrPool on the template patch
        """

        reg_min = self.grid_to_template_x[0][0]
        reg_max = self.grid_to_template_x[-1][-1]

        bbox = np.array(bbox, np.float32)
        bbox = np.clip(bbox, a_max=reg_max, a_min=reg_min)

        sz = 2 * (self.tf_size // 2)
        slope = sz / (reg_max - reg_min)

        return (bbox - reg_min) * slope

    def pool_label_search(self, bbox):
        """
        Get pseudo bbox for PrPool on the search area
        """

        reg_min = self.search_area_x_axis[0]
        reg_max = self.search_area_x_axis[-1]

        bbox = np.array(bbox, np.float32)
        bbox = np.clip(bbox, a_max=reg_max, a_min=reg_min)

        sz = 2 * (self.sf_size // 2)
        slope = sz / (reg_max - reg_min)

        return (bbox - reg_min) * slope

    def _posNegRandom(self):
        """
        Get a random number from [-1, 1]
        """
        return random.random() * 2 - 1.0

    def _toBBox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.template_size

        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def _crop_hwc(self, image, bbox, out_sz, padding=(0, 0, 0)):
        """
        Crop image
        """
        bbox = [float(x) for x in bbox]
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        return crop

    def _draw(self, image, box, name):
        """
        Draw loaded image for debugging
        """
        draw_image = np.array(image.copy()[:, :, ::-1])
        if box is not None:
            x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
            draw_image = draw_image
            cv2.rectangle(draw_image, (x1, y1), (x2, y2), (255, 215, 0), 2)
            # cv2.circle(draw_image, (int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)), 3, (0, 0, 255))
            # cv2.putText(draw_image, '[x: {}, y: {}]'.format(int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)),
            #            (int(round(x1 + x2) / 2) - 3, int(round(y1 + y2) / 2) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
            #            (255, 255, 255), 1)
        cv2.imwrite(name, draw_image[:, :, ::-1])

    def clip_number(self, num, _max=127.0, _min=0.0):

        if num >= _max:
            return _max
        elif num <= _min:
            return _min
        else:
            return num

    def _augmentation(self, image, bbox, size, search=False, cycle_memory=False):
        """
        Data augmentation for input frames
        """
        shape = image.shape
        crop_bbox = center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        if not search:
            # The shift and scale for template
            param.shift = (self._posNegRandom() * self.shift, self._posNegRandom() * self.shift)  # shift
            param.scale = (
                (1.0 + self._posNegRandom() * self.scale), (1.0 + self._posNegRandom() * self.scale))  # scale change
        elif not cycle_memory:
            # The shift and scale for search area
            param.shift = (self._posNegRandom() * self.shift_s, self._posNegRandom() * self.shift_s)  # shift
            param.scale = (
                (1.0 + self._posNegRandom() * self.scale_s),
                (1.0 + self._posNegRandom() * self.scale_s))  # scale change
        else:
            # The shift and scale for memory search areas
            param.shift = (self._posNegRandom() * self.shift_m, self._posNegRandom() * self.shift_m)  # shift
            param.scale = (
                (1.0 + self._posNegRandom() * self.scale_m),
                (1.0 + self._posNegRandom() * self.scale_m))  # scale change

        crop_bbox, _ = aug_apply(Corner(*crop_bbox), param, shape)

        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = BBox(bbox.x1 - x1, bbox.y1 - y1, bbox.x2 - x1, bbox.y2 - y1)

        scale_x, scale_y = param.scale
        bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale

        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=bbox.x1, y1=bbox.y1, x2=bbox.x2, y2=bbox.y2),
        ], shape=image.shape)

        if not search:
            # Augmentation for template
            image, bbs_aug = self.template_aug_seq(image=image, bounding_boxes=bbs)
        elif not cycle_memory:
            # Augmentation for search area
            image, bbs_aug = self.search_aug_seq(image=image, bounding_boxes=bbs)
        else:
            # Augmentation for memory search areas
            image, bbs_aug = self.memory_aug_seq(image=image, bounding_boxes=bbs)

        bbox = Corner(self.clip_number(bbs_aug[0].x1, _max=image.shape[0]),
                      self.clip_number(bbs_aug[0].y1, _max=image.shape[1]),
                      self.clip_number(bbs_aug[0].x2, _max=image.shape[0]),
                      self.clip_number(bbs_aug[0].y2, _max=image.shape[1]))

        return image, bbox, param

    def _dynamic_label(self, fixedLabelSize, c_shift, rPos=2, rNeg=0):
        """
        Generating classification label
        """
        if isinstance(fixedLabelSize, int):
            fixedLabelSize = [fixedLabelSize, fixedLabelSize]

        assert (fixedLabelSize[0] % 2 == 1)

        d_label = self._create_dynamic_logisticloss_label(fixedLabelSize, c_shift, rPos, rNeg)

        return d_label

    def _create_dynamic_logisticloss_label(self, label_size, c_shift, rPos=2, rNeg=0):
        if isinstance(label_size, int):
            sz = label_size
        else:
            sz = label_size[0]

        sz_x = sz // 2 + int(-c_shift[0] / 8)  # 8 is the stride
        sz_y = sz // 2 + int(-c_shift[1] / 8)

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        dist_to_center = np.abs(x) + np.abs(y)  # Block metric
        label = np.where(dist_to_center <= rPos,
                         np.ones_like(y),
                         np.where(dist_to_center < rNeg,
                                  0.5 * np.ones_like(y),
                                  np.zeros_like(y)))
        return label


class subData(object):
    """
    Sub dataset class for training USOT with multi dataset
    """

    def __init__(self, cfg, data_name, start, memory_num, video_quality, far_sample):
        self.data_name = data_name
        self.start = start

        # Dataset info
        info = cfg.USOT.DATASET[data_name]
        self.root = info.PATH

        with open(info.ANNOTATION) as fin:
            self.labels = json.load(fin)
            self._clean()
            # Video number
            self.num = len(self.labels)

        # Number of training instances used in each epoch for a certain dataset
        self.num_use = info.USE
        # Number of memory frames in a single training instance
        self.memory_num = memory_num
        # The threshold to filter videos
        self.video_quality = video_quality
        # When sampling memory frames, first sample (memory_num + far_sample) frames in the video fragment,
        #             and then pick (memory_num) frames "most far from" the template frame
        self.far_sample = far_sample

        self._shuffle()

    def _clean(self):
        """
        Remove empty videos/frames/annos in dataset
        """
        # No frames
        to_del = []
        for video in self.labels:
            frames = self.labels[video]
            if len(frames) <= 0:
                print("warning {} has no frames.".format(video))
                to_del.append(video)

        for video in to_del:
            try:
                del self.labels[video]
            except:
                pass

        print(self.data_name)

        self.videos = list(self.labels.keys())
        print('{} loaded.'.format(self.data_name))

    def _shuffle(self):
        """
        Shuffle to get random pairs index (video)
        """
        lists = list(range(self.start, self.start + self.num))
        m = 0
        pick = []
        while m < self.num_use:
            sample_random.shuffle(lists)
            pick += lists
            m += self.num

        self.pick = pick[:self.num_use]
        return self.pick

    def _calc_video_quality(self, bbox_picked_freq, corner_bbox_freq):
        """
        The function to calculate video quality with DP-selection frequency in video
        In practice, we additionally give penalty to video sequences with lots of pseudo boxes lying at the corner
        """

        return bbox_picked_freq - 1 / 3 * corner_bbox_freq

    def _calc_short_term_frame_quality(self, bbox_info):
        """
        The function to calculate short-term frame quality for sampling template frames for naive Siamese tracker
        bbox_info structure:
             index 0-3: [x1, y1, x2, y2]
             index 4-5: [short-term DP-pick-freq, long-term DP-pick-freq] for a single frame
             index 6-8: [T_l, T_u, corner_score]
        We use short-term DP-pick-freq as the basic frame quality, and use corner_score to refine it.
        As an implementation detail, we give penalty to frames with their pseudo boxes lying at the corner
        """

        return bbox_info[4] + 2 / 3 * bbox_info[8]

    def _calc_long_term_frame_quality(self, bbox_info, video_len):
        """
        The function to calculate long-term frame quality for sampling template frames for cycle memory training
        bbox_info structure:
             index 0-3: [x1, y1, x2, y2]
             index 4-5: [short-term DP-pick-freq, long-term DP-pick-freq] for a single frame
             index 6-8: [T_l, T_u, corner_score]
        We use short-term DP-pick-freq as the basic frame quality, and use T_u, T_l, and corner_score to refine it.
        As an implementation detail, we give penalty to frames with their pseudo boxes lying at the corner.
        For cycle memory, we also give priority to template frames with higher (T_u - T_l) frame intervals
        """

        return bbox_info[4] + 1 / 2 * bbox_info[8] + (bbox_info[7] - bbox_info[6]) / (video_len * 2)

    def _get_siamese_image_anno(self, video, track_id, video_index=None):
        """
        Loader logic for naive Siamese training
        Sampling the template frame and obtaining its pseudo annotation
        """
        video_info = self.labels[video]
        track_info = video_info[track_id]

        # Threshold to pick reliable videos
        video_tolerance_threshold = self.video_quality

        # Step 1: testify the currently picked video by video quality score
        bbox_picked_freq = track_info['meta']['bbox_picked_freq']
        corner_bbox_freq = track_info['meta']['corner_bbox_freq']
        video_quality_score = self._calc_video_quality(bbox_picked_freq, corner_bbox_freq)

        # Sample more candidate frames if the video quality is lower, and vice versa
        frame_candidate_num = int((1.0 / bbox_picked_freq) * 3)

        # Branch 1
        # If the picked video is of high quality to be used as training data, simply pick the most reliable frame
        if video_quality_score >= video_tolerance_threshold and corner_bbox_freq < 0.25:
            frames = list(track_info.keys())
            if 'meta' in frames:
                frames.remove('meta')
            video_len = len(frames)
            picked_frame_candidates_s = np.random.choice(video_len, frame_candidate_num, replace=True)

            # Calculate short-term frame quality
            short_term_frame_quality_s = np.array([self._calc_short_term_frame_quality(track_info[frames[cand]])
                                                   for cand in picked_frame_candidates_s])
            # Select the frame with the highest frame quality
            max_cand_index_s = np.argmax(short_term_frame_quality_s)
            max_cand_frame_s = picked_frame_candidates_s[max_cand_index_s]

            frame_id_s = frames[int(max_cand_frame_s)]
            frame_id_s_format = '0' * (8 - len(frame_id_s)) + frame_id_s
            image_path_s = join(self.root, video, "{}.{}.x.jpg".format(frame_id_s_format[-6:], track_id))

            # Return the single frame for template-search pair
            return image_path_s, track_info[frame_id_s][:4]

        # Branch 2
        # If the picked video is not of high quality, re-sample video from its nearby videos
        # Step 2: re-sample video for the original randomly sampled video is of low quality
        video_total_num = len(self.labels)
        candidate_range = np.arange(max(0, video_index - 30), min(video_total_num - 1, video_index + 31))

        # Sample another video from nearby videos, and pick the video with the highest quality score
        max_pick_times = 20
        video_candidate_num = 3
        max_freq_video = None
        track_id = None
        while max_pick_times:
            picked_candidates = np.random.choice(candidate_range, video_candidate_num, replace=True)
            picked_candidates_video_name = [self.videos[cand] for cand in picked_candidates]
            picked_track_id = [random.choice(list(self.labels[video_name].keys()))
                               for video_name in picked_candidates_video_name]
            video_scores = np.array([self._calc_video_quality(
                self.labels[picked_candidates_video_name[cand_ind]][picked_track_id[cand_ind]]['meta'][
                    'bbox_picked_freq'],
                self.labels[picked_candidates_video_name[cand_ind]][picked_track_id[cand_ind]]['meta'][
                    'corner_bbox_freq'])
                for cand_ind in range(len(picked_candidates_video_name))])
            max_freq_index = np.argmax(video_scores)
            max_freq_video = picked_candidates[max_freq_index]
            track_id = picked_track_id[max_freq_index]

            # Check if the currently selected video is of high quality or not
            if video_scores[max_freq_index] > video_tolerance_threshold:
                break
            else:
                max_pick_times -= 1

        # Extreme case: if no video is determined even after 20 trials, then randomly pick one.
        if max_freq_video is None or track_id is None:
            max_freq_video = np.random.choice(candidate_range, 1)
            track_id = random.choice(list(self.labels[self.videos[max_freq_video]].keys()))

        # Re-sampling video finished
        video = self.videos[max_freq_video]
        video_info = self.labels[video]
        track_info = video_info[track_id]
        bbox_picked_freq = track_info['meta']['bbox_picked_freq']
        frame_candidate_num = int((1.0 / bbox_picked_freq) * 3)

        # Step 3: re-sample frames according to the frame quality
        frames = list(track_info.keys())
        if 'meta' in frames:
            frames.remove('meta')
        video_len = len(frames)
        picked_frame_candidates_s = np.random.choice(video_len, frame_candidate_num, replace=True)

        # Calculate short-term frame quality
        short_term_frame_quality_s = np.array([self._calc_short_term_frame_quality(track_info[frames[cand]])
                                               for cand in picked_frame_candidates_s])
        # Select the frame with the highest frame quality
        max_cand_index_s = np.argmax(short_term_frame_quality_s)
        max_cand_frame_s = picked_frame_candidates_s[max_cand_index_s]

        frame_id_s = frames[int(max_cand_frame_s)]
        frame_id_s_format = '0' * (8 - len(frame_id_s)) + frame_id_s
        image_path_s = join(self.root, video, "{}.{}.x.jpg".format(frame_id_s_format[-6:], track_id))

        # Return the single frame for template-search pair
        return image_path_s, track_info[frame_id_s][:4]

    def _get_cycle_memory_image_anno(self, video, track_id, video_index=None):

        """
        Loader logic for cycle memory training
        Sampling the template frame (with pseudo annotation) as well as N_mem memory frames
        """
        video_info = self.labels[video]
        track_info = video_info[track_id]

        # Threshold to pick reliable videos
        video_tolerance_threshold = self.video_quality

        # Step 1: test the currently picked video
        bbox_picked_freq = track_info['meta']['bbox_picked_freq']
        corner_bbox_freq = track_info['meta']['corner_bbox_freq']
        video_quality_score = self._calc_video_quality(bbox_picked_freq, corner_bbox_freq)

        # Sample more candidate frames if the video quality is lower, and vice versa
        frame_candidate_num = int((1.0 / bbox_picked_freq) * 3)

        # Branch 1
        # If the picked video is of high quality to be used as training data, simply pick the most reliable frame
        if video_quality_score >= video_tolerance_threshold and corner_bbox_freq < 0.25:
            frames = list(track_info.keys())
            if 'meta' in frames:
                frames.remove('meta')
            video_len = len(frames)
            picked_frame_candidates_s = np.random.choice(video_len, frame_candidate_num, replace=True)

            # Note that long-term frame quality is slightly different from short-term frame quality
            # For cycle memory, we also give priority to template frames with higher (T_u - T_l) frame intervals
            long_term_frame_quality_s = np.array(
                [self._calc_long_term_frame_quality(track_info[frames[cand]], video_len)
                 for cand in picked_frame_candidates_s])
            max_cand_index_s = np.argmax(long_term_frame_quality_s)
            max_cand_frame_s = int(picked_frame_candidates_s[max_cand_index_s])

            frame_id_s = frames[max_cand_frame_s]
            frame_id_s_format = '0' * (8 - len(frame_id_s)) + frame_id_s
            image_path_s = join(self.root, video, "{}.{}.x.jpg".format(frame_id_s_format[-6:], track_id))

            # Now begin to sample memory frames from nearby frames of the template frame
            search_range = np.arange(track_info[frame_id_s][6], track_info[frame_id_s][7] + 1)
            # First sample (memory_num + far_sample) frames in video fragment determined by DP
            picked_frame_nearby_s = np.random.choice(search_range, self.memory_num + self.far_sample, replace=True)
            interval_abs = np.abs(picked_frame_nearby_s - max_cand_frame_s)
            # Pick memory_num frames "most far from" the template frame (somewhat like hard negative mining?)
            select_idx = interval_abs.argsort()[::-1][0:self.memory_num]
            picked_frame_nearby_s = picked_frame_nearby_s[select_idx]

            # Uncomment here to do statistics for averaged frame interval
            # print(len(search_range)-1)

            frame_id_nearby_s = [frames[int(cand)] for cand in picked_frame_nearby_s]
            frame_id_nearby_s_format = ['0' * (8 - len(frame_id)) + frame_id for frame_id in frame_id_nearby_s]
            image_path_nearby_s = [join(self.root, video, "{}.{}.x.jpg".format(frame_id[-6:], track_id))
                                   for frame_id in frame_id_nearby_s_format]
            bbox_nearby_s = [track_info[frame_id][:4] for frame_id in frame_id_nearby_s]

            # Return template frame and memory frames
            return image_path_s, track_info[frame_id_s][:4], image_path_nearby_s, bbox_nearby_s

        # Branch 2
        # If the picked video is not of high quality, sample video from its nearby videos
        # Step 2: re-sample video
        video_total_num = len(self.labels)
        candidate_range = np.arange(max(0, video_index - 30), min(video_total_num - 1, video_index + 31))

        # Sample from nearby videos, and pick the video with the highest quality score
        max_pick_times = 20
        video_candidate_num = 3
        max_quality_video = None
        track_id = None
        while max_pick_times:
            picked_candidates = np.random.choice(candidate_range, video_candidate_num, replace=True)
            picked_candidates_video_name = [self.videos[cand] for cand in picked_candidates]
            picked_track_id = [random.choice(list(self.labels[video_name].keys()))
                               for video_name in picked_candidates_video_name]
            video_quality_scores = np.array([self._calc_video_quality(
                self.labels[picked_candidates_video_name[cand_ind]][picked_track_id[cand_ind]]['meta'][
                    'bbox_picked_freq'],
                self.labels[picked_candidates_video_name[cand_ind]][picked_track_id[cand_ind]]['meta'][
                    'corner_bbox_freq'])
                for cand_ind in range(len(picked_candidates_video_name))])
            max_quality_index = np.argmax(video_quality_scores)
            max_quality_video = picked_candidates[max_quality_index]
            track_id = picked_track_id[max_quality_index]

            # Check if the currently selected video is of high quality or not
            if video_quality_scores[max_quality_index] > video_tolerance_threshold:
                break
            else:
                max_pick_times -= 1

        # Extreme case: if no video is picked even after 20 trials, then randomly pick one.
        if max_quality_video is None or track_id is None:
            max_quality_video = np.random.choice(candidate_range, 1)
            track_id = random.choice(list(self.labels[self.videos[max_quality_video]].keys()))

        # Re-sampling video finished
        video = self.videos[max_quality_video]
        video_info = self.labels[video]
        track_info = video_info[track_id]
        bbox_picked_freq = track_info['meta']['bbox_picked_freq']
        frame_candidate_num = int((1.0 / bbox_picked_freq) * 3)

        # Step 3: re-sample frames
        frames = list(track_info.keys())
        if 'meta' in frames:
            frames.remove('meta')
        video_len = len(frames)
        picked_frame_candidates_s = np.random.choice(video_len, frame_candidate_num, replace=True)

        # Note that long-term frame quality is slightly different from short-term frame quality
        # For cycle memory, we also give priority to template frames with higher (T_u - T_l) frame intervals
        long_term_frame_quality_s = np.array([self._calc_long_term_frame_quality(track_info[frames[cand]], video_len)
                                              for cand in picked_frame_candidates_s])
        max_cand_index_s = np.argmax(long_term_frame_quality_s)
        max_cand_frame_s = picked_frame_candidates_s[max_cand_index_s]

        frame_id_s = frames[int(max_cand_frame_s)]
        frame_id_s_format = '0' * (8 - len(frame_id_s)) + frame_id_s
        image_path_s = join(self.root, video, "{}.{}.x.jpg".format(frame_id_s_format[-6:], track_id))

        # Now begin to sample memory frames from nearby frames of the template frame
        search_range = np.arange(track_info[frame_id_s][6], track_info[frame_id_s][7] + 1)
        # First sample (memory_num + far_sample) frames in video fragment determined by DP
        picked_frame_nearby_s = np.random.choice(search_range, self.memory_num + self.far_sample, replace=True)

        interval_abs = np.abs(picked_frame_nearby_s - max_cand_frame_s)
        # Pick memory_num frames "most far from" the template frame (somewhat like hard negative mining?)
        select_idx = interval_abs.argsort()[::-1][0:self.memory_num]
        picked_frame_nearby_s = picked_frame_nearby_s[select_idx]

        # Uncomment here to do statistics for frame interval
        # print(len(search_range)-1)

        frame_id_nearby_s = [frames[int(cand)] for cand in picked_frame_nearby_s]
        frame_id_nearby_s_format = ['0' * (8 - len(frame_id)) + frame_id for frame_id in frame_id_nearby_s]
        image_path_nearby_s = [join(self.root, video, "{}.{}.x.jpg".format(frame_id[-6:], track_id))
                               for frame_id in frame_id_nearby_s_format]
        bbox_nearby_s = [track_info[frame_id][:4] for frame_id in frame_id_nearby_s]

        # Return the template frame and memory frames
        return image_path_s, track_info[frame_id_s][:4], image_path_nearby_s, bbox_nearby_s

    def _get_instances(self, index, cycle_memory=False):
        """
        get training instances
        """
        video_name = self.videos[index]
        video = self.labels[video_name]
        track_id = random.choice(list(video.keys()))

        if cycle_memory:
            # For cycle memory training (returning one search frame and several memory frames)
            return self._get_cycle_memory_image_anno(video_name, track_id, video_index=index)
        else:
            # For offline naive Siamese tracker (one template and one search area picked in the same frame)
            return self._get_siamese_image_anno(video_name, track_id, video_index=index)


if __name__ == '__main__':

    import os
    from torch.utils.data import DataLoader
    from lib.config.config_usot import config

    # For testing and visualizing dataloader, you can set self.loader_test to True
    # Then example training instances can be found in $USOT_PATH/var/loader
    vis_dataloader_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       "..", "..", "var", "loader")
    if not os.path.exists(vis_dataloader_path):
        os.makedirs(vis_dataloader_path)

    # Cycle the dataloader
    train_set = USOTDataset(config)
    train_set.loader_test = True
    train_loader = DataLoader(train_set, batch_size=4, num_workers=1, pin_memory=False)

    for iter, input in enumerate(train_loader):
        template = input[0]
        search = input[1]
