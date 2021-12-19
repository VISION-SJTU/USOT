# -----------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Revised by Jilai Zheng, for USOT
# ------------------------------------------------------------------------------
from lib.models.connect import AdjustLayer, box_tower_reg
from lib.models.backbones import ResNet50
from lib.models.prroi_pool.functional import prroi_pool2d
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class USOT_(nn.Module):
    def __init__(self, mem_size=4, pr_pool=True,
                 search_size=255, score_size=25, maximum_batch=16, sf_size=25):
        super(USOT_, self).__init__()
        self.features = None
        self.connect_model = None
        self.zf = None
        self.criterion = nn.BCEWithLogitsLoss()
        self.neck = None
        self.search_size = search_size
        self.score_size = score_size
        self.search_feature_size = sf_size

        # Param maximum batch is only used for generating a valid grid, and does not effect the actual batch size
        self.maximum_batch = maximum_batch if self.training else 1
        # Number of memory frames
        self.mem_size = mem_size

        # Always keep pr_pool = True for training a bbox regression module
        self.pr_pool = pr_pool

        self.grids()

    def feature_extractor(self, x):
        return self.features(x)

    def _cls_loss(self, pred, label, select):
        if len(select.size()) == 0:
            return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.criterion(pred, label)

    def _weighted_BCE(self, pred, label):
        pred = pred.view(-1)
        label = label.view(-1)
        pos = label.data.eq(1).nonzero().squeeze().cuda()
        neg = label.data.eq(0).nonzero().squeeze().cuda()

        loss_pos = self._cls_loss(pred, label, pos)
        loss_neg = self._cls_loss(pred, label, neg)

        return loss_pos * 0.5 + loss_neg * 0.5

    def _IOULoss(self, pred, target):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        assert losses.numel() != 0
        return losses.mean()

    def add_iouloss(self, bbox_pred, reg_target, reg_weight):
        """
        Add IoU Loss for bbox regression
        """

        bbox_pred_flatten = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        reg_target_flatten = reg_target.reshape(-1, 4)
        reg_weight_flatten = reg_weight.reshape(-1)
        pos_inds = torch.nonzero(reg_weight_flatten > 0).squeeze(1)

        bbox_pred_flatten = bbox_pred_flatten[pos_inds]
        reg_target_flatten = reg_target_flatten[pos_inds]

        loss = self._IOULoss(bbox_pred_flatten, reg_target_flatten)

        return loss

    def grids(self):
        """
        Each element of feature map on response map
        :return: H*W*2 (position for each element)
        """
        # Grid for response map
        sz = self.score_size
        stride = 8
        sz_x = sz // 2
        sz_y = sz // 2
        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search = {}
        self.grid_to_search_x = x * stride + self.search_size // 2
        self.grid_to_search_y = y * stride + self.search_size // 2

        self.grid_to_search_x = torch.Tensor(self.grid_to_search_x).unsqueeze(0).unsqueeze(0).cuda()
        self.grid_to_search_y = torch.Tensor(self.grid_to_search_y).unsqueeze(0).unsqueeze(0).cuda()

        self.grid_to_search_x = self.grid_to_search_x.repeat(self.maximum_batch * self.mem_size, 1, 1, 1)
        self.grid_to_search_y = self.grid_to_search_y.repeat(self.maximum_batch * self.mem_size, 1, 1, 1)

        # Axis for search area feature
        sz = self.search_feature_size
        stride = 8
        sz_x = sz // 2
        self.search_area_x_axis = (np.arange(0, sz) - np.floor(float(sz_x))) * stride + self.search_size // 2

    def pred_offset_to_image_bbox(self, bbox_pred, batch):
        """
        Convert bbox from the predicted response map axis to the image-level axis
        """

        self.grid_to_search_x = self.grid_to_search_x[0:batch * self.mem_size].to(bbox_pred.device)
        self.grid_to_search_y = self.grid_to_search_y[0:batch * self.mem_size].to(bbox_pred.device)

        pred_x1 = self.grid_to_search_x - bbox_pred[:, 0, ...].unsqueeze(1)
        pred_y1 = self.grid_to_search_y - bbox_pred[:, 1, ...].unsqueeze(1)
        pred_x2 = self.grid_to_search_x + bbox_pred[:, 2, ...].unsqueeze(1)
        pred_y2 = self.grid_to_search_y + bbox_pred[:, 3, ...].unsqueeze(1)

        pred = [pred_x1, pred_y1, pred_x2, pred_y2]

        pred = torch.cat(pred, dim=1)

        return pred

    def image_bbox_to_prpool_bbox(self, image_bbox):
        """
        Convert bbox from the image axis to the search area axis
        """

        reg_min = self.search_area_x_axis[0]
        reg_max = self.search_area_x_axis[-1]
        sz = 2 * (self.search_feature_size // 2)
        gap = (reg_max - reg_min) / sz
        image_bbox = torch.clamp(image_bbox, max=reg_max + 2 * gap, min=reg_min - 2 * gap)

        slope = 1.0 / gap
        return (image_bbox - reg_min) * slope

    def prpool_feature(self, features, bboxs):
        """
        PrPool from deep features according to specific bboxes
        """

        batch_index = torch.arange(0, features.shape[0]).view(-1, 1).float().to(features.device)
        bboxs_index = torch.cat((batch_index, bboxs), dim=1)
        return prroi_pool2d(features, bboxs_index, 7, 7, 1.0)

    def template(self, z, template_bbox=None):
        _, self.zf = self.feature_extractor(z)

        if self.neck is not None:
            _, self.zf = self.neck(self.zf, crop=True, pr_pool=self.pr_pool, bbox=template_bbox)

    def track(self, x, template_mem=None, score_mem=None):

        _, xf = self.feature_extractor(x)

        if self.neck is not None:
            xf = self.neck(xf)

        if template_mem is not None:
            # Track with both offline and online module (with memory queue features existing)
            bbox_pred, cls_pred, cls_feature, reg_feature, cls_memory_pred = self.connect_model(xf, kernel=self.zf,
                                                                                                memory_kernel=template_mem,
                                                                                                memory_confidence=score_mem)

            # Here xf is the feature of search areas which will be cropped soon according to the final bbox
            return cls_pred, bbox_pred, cls_memory_pred, xf
        else:
            # Track with offline module only
            bbox_pred, cls_pred, _, _, _ = self.connect_model(xf, kernel=self.zf)

            return cls_pred, bbox_pred, None, None

    def extract_memory_feature(self, ori_x=None, xf=None, search_bbox=None):
        # Note that here search bbox is the bbox on the deep feature (not on the original search frame)
        if ori_x is not None:
            _, xf = self.feature_extractor(ori_x)
            xf = self.neck(xf, crop=False)
        features = self.prpool_feature(xf, search_bbox)
        return features

    def forward(self, template, search, label=None, reg_target=None,
                reg_weight=None, template_bbox=None, search_memory=None,
                search_bbox=None, cls_ratio=0.40):
        """
        Training pipeline for both naive Siamese and cycle memory
        """
        # Feature extraction for template patch and search area
        _, zf = self.feature_extractor(template)
        _, xf = self.feature_extractor(search)

        # PrPool the template feature and down-sample the deep features
        if self.neck is not None:
            _, zf = self.neck(zf, crop=True, pr_pool=self.pr_pool, bbox=template_bbox)
            xf = self.neck(xf, crop=False)

        if search_memory is not None:
            # Original Siamese fg/bg cls and bbox reg branch (self-track in paper)
            bbox_pred, cls_pred, cls_x, _, _ = self.connect_model(xf, kernel=zf)

            # Add bbox regression loss
            reg_loss = self.add_iouloss(bbox_pred, reg_target, reg_weight)
            # Add naive Siamese cls loss
            cls_loss_ori = self._weighted_BCE(cls_pred, label)

            # Now begin to calculate cycle memory loss
            # Extract deep features for memory search areas
            batch, mem_size, cx, hx, wx = search_memory.shape
            search_memory = search_memory.view(-1, cx, hx, wx)
            _, xf_mem = self.feature_extractor(search_memory)
            xf_mem = self.neck(xf_mem, crop=False)

            # Prepare feature for mem_forward_cls (forward tracking with the online module)
            search_pooled_feature = self.prpool_feature(xf, search_bbox)
            batch, cspf, wspf, hspf = search_pooled_feature.shape
            search_pooled_feature = search_pooled_feature.view(batch, 1, cspf, wspf, hspf)
            search_pooled_feature = search_pooled_feature.repeat(1, mem_size, 1, 1, 1)
            search_pooled_feature = search_pooled_feature.view(-1, cspf, wspf, hspf)

            # Repeat the original template
            batch, cz, hz, wz = zf.shape
            zf_mem = zf.view(batch, 1, cz, hz, wz)
            zf_mem = zf_mem.repeat(1, mem_size, 1, 1, 1)
            zf_mem = zf_mem.view(-1, cz, hz, wz)

            # Get the intermediate target bbox and cls score in memory search areas (tracking with offline module)
            off_forward_bbox, off_forward_cls, forward_x_store, _, _ = self.connect_model(xf_mem, kernel=zf_mem)

            # Get the mem_forward_cls score in memory search areas (tracking with online module)
            fake_confidence = torch.ones(batch * mem_size, 1)
            _, _, _, _, mem_forward_cls = self.connect_model(xf_mem, memory_kernel=search_pooled_feature,
                                                             memory_confidence=fake_confidence,
                                                             cls_x_store=forward_x_store)
            mem_forward_cls = mem_forward_cls.view(batch, mem_size, -1)
            off_forward_cls = off_forward_cls.view(batch, mem_size, -1)

            # Linearly combine off_forward_cls and mem_forward_cls as the forward response map
            # Note: weighted add memory_forward_cls and off_forward_cls, while bbox remains
            forward_res_map = cls_ratio * off_forward_cls + (1 - cls_ratio) * mem_forward_cls
            best_forward_cls = forward_res_map.max(dim=2)
            best_forward_cls_argmax = best_forward_cls.indices.view(batch, mem_size, 1, 1)
            best_forward_cls_argmax = best_forward_cls_argmax.repeat(1, 1, 1, 4)
            bbox_pred_to_img = self.pred_offset_to_image_bbox(off_forward_bbox, batch)
            bbox_pred_to_img = bbox_pred_to_img.view(batch, mem_size, 4, -1).transpose(2, 3)
            best_mem_bbox = torch.gather(bbox_pred_to_img, dim=2,
                                         index=best_forward_cls_argmax).view(batch * mem_size, 4)
            best_forward_cls_score = best_forward_cls.values.detach()
            best_forward_bbox_pool = self.image_bbox_to_prpool_bbox(best_mem_bbox).detach()

            # PrPool intermediate features from memory search areas as the memory queue
            pooled_mem_features = self.prpool_feature(xf_mem, best_forward_bbox_pool)
            # Backward track from memory queue to the search area in the template frame
            _, _, _, _, backward_res_map = self.connect_model(xf, memory_kernel=pooled_mem_features,
                                                              memory_confidence=best_forward_cls_score,
                                                              cls_x_store=cls_x)

            # Cycle memory loss is calculated with the same pseudo label as original cls loss
            cls_memory_loss = self._weighted_BCE(backward_res_map, label)

            return cls_loss_ori, cls_memory_loss, reg_loss

        else:
            # The following logic is for purely offline naive Siamese training
            bbox_pred, cls_pred, _, _, _ = self.connect_model(xf, kernel=zf)

            cls_loss = self._weighted_BCE(cls_pred, label)
            reg_loss = self.add_iouloss(bbox_pred, reg_target, reg_weight)

            return cls_loss, None, reg_loss


class USOT(USOT_):
    def __init__(self, settings=None):
        if settings is None:
            settings = {'mem_size': 4, 'pr_pool': True}
        super(USOT, self).__init__(mem_size=settings['mem_size'], pr_pool=settings['pr_pool'],
                                   search_size=255, score_size=25, maximum_batch=16, sf_size=25)
        self.features = ResNet50(used_layers=[3])  # in param
        self.neck = AdjustLayer(in_channels=1024, out_channels=256, pr_pool=settings['pr_pool'])
        self.connect_model = box_tower_reg(in_channels=256, out_channels=256, tower_num=4)
