import imageio
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import torch
import cv2
from easydict import EasyDict
from torchvision import transforms
from preprocessing.flow_module.transforms import Zoom, ArrayToTensor

from preprocessing.flow_module.flow_utils import flow_to_image, calc_corner_bbox_freq, \
    resize_flow, flow_to_bbox, restore_model, smooth_bbox_dp, calc_nearby_bbox_freq
from preprocessing.flow_module.models.pwclite import PWCLite


# The test helper for ARFlow
class TestHelper:
    def __init__(self, cfg):
        self.cfg = EasyDict(cfg)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        self.model = self.init_model()
        self.input_transform = transforms.Compose([
            Zoom(*self.cfg.test_shape),
            ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        ])

    # Init model for ARFlow
    def init_model(self):

        model = PWCLite(self.cfg.model)
        # print('Number fo parameters: {}'.format(model.num_parameters()))
        model = model.to(self.device)
        model = restore_model(model, self.cfg.pretrained_model)
        model.eval()
        return model

    # Run single image instances
    def run(self, imgs):
        imgs = [self.input_transform(img).unsqueeze(0) for img in imgs]
        img_pair = torch.cat(imgs, 1).to(self.device)
        return self.model(img_pair)

    # Run for images in a whole video sequence
    def run_sequence(self, imgs, size, gap=3, init_adjacent=4):

        imgs = [self.input_transform(img).unsqueeze(0).to(self.device) for img in imgs]
        flows = []
        # Here introduces a trick of determining the frame interval for estimating optical flow
        # This interval is indicated as T_f in paper, and as variable 'adjacent' in the following codes
        # Basic logic: the interval for estimating flow should decline if the image is flowing too fast, and vice versa
        # We actually init the interval as 4, and it will fluctuate between 1 and 7 according to the estimated flow
        adjacent = init_adjacent

        # Also note here that we actually estimate flow map and sample candidate boxes on sub-sampled videos
        # i.e. estimate flow map every 'gap' frames, by default the param 'gap' is chosen as 3
        for i in range(gap, len(imgs) - gap, gap):

            # Variable 'direction': 0 for borderline, -1 for down, +1 for up
            # For each frame, param adjacent can only be switched either upward or downward
            direction = 0
            while True:
                # In fact, ARFlow uses 3 frames as input, namely frames t-T_f, t and t+T_f
                min_index = max(0, i-adjacent)
                max_index = min(i+adjacent, len(imgs)-1)
                imgs_cat = [imgs[min_index], imgs[i], imgs[max_index]]
                img_pair = torch.cat(imgs_cat, 1)
                # Use ARFlow to estimate optical flow
                flow = self.model(img_pair)
                # We only collect forward flow (from frame t to frame t+T_f) from ARFlow results (flow_fw)
                flow = flow['flows_fw'][0]
                flow = resize_flow(flow, size)
                flow = flow[0].detach().cpu().numpy().transpose([1, 2, 0])

                # print("max: {}, mean: {}, min: {}.".format(flow.max(), flow.mean(), flow.min()))
                abs_max = max(abs(flow.max()), abs(flow.min()))

                # Param 'adjacent' (T_f, frame interval for flow estimating) declines if the image is flowing too fast
                if abs_max > 16 and adjacent >= 2 and direction <= 0:
                    adjacent -= 1
                    direction = -1
                # Param 'adjacent' (T_f, frame interval for flow estimating) increases if the image is flowing too slow
                elif abs_max < 8 and adjacent <= 6 and direction >= 0:
                    adjacent += 1
                    direction = 1
                else:
                    break

            # print("frame {} has param adjacent {}.".format(i, adjacent))
            flows.append(flow)
        return flows


# Init the flow estimation module
def init_module(model=None, test_shape=None):
    if test_shape is None:
        test_shape = [384, 640]
    if model is None:
        model = os.path.join(os.path.dirname(__file__), "checkpoint", "pwclite_ar_mv.tar")

    cfg = {
        'model': {
            'upsample': True,
            'n_frames': 3,
            'reduce_dense': True
        },
        'pretrained_model': model,
        'test_shape': test_shape,
    }
    ts = TestHelper(cfg)
    return ts


def inference_sequence(ts, image_list, vis=True, gap=3, init_adjacent=4):

    # Load all frames in a video sequence
    imgs = [imageio.imread(img).astype(np.float32) for img in image_list]
    h, w = imgs[0].shape[:2]

    # Estimating optical flow for the whole video sequence
    # Note that we actually estimate flow maps and sample candidate boxes on sub-sampled videos
    # i.e. estimate flow map every 'gap' frames, by default the param 'gap' is chosen as 3 (except YT-VOS)
    flows = ts.run_sequence(imgs, size=(h, w), gap=gap, init_adjacent=init_adjacent)

    # Note that cut_ratio is used to cut the margins of the flow map, as margin flows are always of low quality
    cut_ratio = 1/32
    # Convert flow map to candidate boxes (B)
    bboxs = [flow_to_bbox(flow, cut_ratio=cut_ratio) for flow in flows]
    # Use Dynamic Programming (DP) to generate reliable pseudo box sequences (B')
    bboxs, picked_frame_index, bbox_found_freq, bbox_picked_freq, aver_vary = \
        smooth_bbox_dp(bboxs, length=len(imgs), gap=gap)

    # Calc the bbox DP-select rate (bbox_freq) for every frame among all its adjacent frames
    # Note: search range is the frame interval for calculating frame quality (Denoted as T_s in the paper)
    # In practice, short interval (3) is better according to our experiments (10 is deprecated)
    freq_dict = calc_nearby_bbox_freq(picked_frame_index, video_length=len(bboxs),
                                      search_range=[3, 10], gap=gap)

    # The frequency of corner bboxes in the smoothed bbox sequence (B')
    # As an implementation detail, we actually give priority to sequences with less corner boxes (for center bias)
    corner_bbox_freq = calc_corner_bbox_freq(bboxs, img_shape=flows[0].shape, cut_ratio=cut_ratio)

    # Visualize optical flow
    # flow_vis = [flow_to_image(flow) for flow in flows]

    if vis:
        i = 0
        while True:
            if i >= len(image_list):
                i = 0
            bbox = bboxs[i]
            image = imgs[i].astype(np.uint8)
            image = image[:, :, ::-1].copy()
            text = "{:.2f}/{:.2f}".format(freq_dict[i][0], freq_dict[i][1])
            draw = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                 (0, 255, 0), 1)
            draw = cv2.putText(draw, text, (int(bbox[0])+25, int(bbox[1])+25),
                               cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            i += 1
            cv2.imshow("Frame", draw)
            time.sleep(0.05)
            key = cv2.waitKey(1) & 0xFF
            # If the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

    return bboxs, picked_frame_index, (freq_dict, bbox_found_freq, bbox_picked_freq, aver_vary, corner_bbox_freq)
