# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and Zhipeng Zhang
# Revised by Jilai Zheng, for USOT
# ------------------------------------------------------------------------------


import os
import yaml
from easydict import EasyDict as edict

config = edict()

# ------ Config for general parameters ------
config.GPUS = "0,1,2,3"
config.WORKERS = 32
config.PRINT_FREQ = 10
config.OUTPUT_DIR = 'var/log'
config.CHECKPOINT_DIR = 'var/snapshot'

# #-----————- Config domains for USOT ------------
config.USOT = edict()
config.USOT.TRAIN = edict()
config.USOT.TEST = edict()
config.USOT.DATASET = edict()
config.USOT.DATASET.VID = edict()
config.USOT.DATASET.GOT10K = edict()
config.USOT.DATASET.LASOT = edict()
config.USOT.DATASET.YTVOS = edict()

# Augmentation for template, search area and memory frames
config.USOT.DATASET.SHIFT = 4
config.USOT.DATASET.SCALE = 0.05
config.USOT.DATASET.SHIFTs = 64
config.USOT.DATASET.SCALEs = 0.18
config.USOT.DATASET.SHIFTm = 64
config.USOT.DATASET.SCALEm = 0.18

# Video quality threshold and far sampling param
config.USOT.DATASET.VIDEO_QUALITY = 0.40
config.USOT.DATASET.FAR_SAMPLE = 3

# The following codes are configs for datasets
# Generally speaking, these configs will be replaced with configs in $USOT_PATH/experiments/train/USOT.yaml
# However, these configs will take effect when testing the dataloader offline, see "datasets_usot.py"
# VID configs
config.USOT.DATASET.VID.PATH = '/home/jlzheng/dataset/tracking/usot/VID_flow/crop511/'
config.USOT.DATASET.VID.ANNOTATION = '/home/jlzheng/dataset/tracking/usot/VID_flow/train.json'
config.USOT.DATASET.VID.USE = 14000

# GOT-10k configs
config.USOT.DATASET.GOT10K.PATH = '/home/jlzheng/dataset/tracking/usot/got10k_flow/crop511/'
config.USOT.DATASET.GOT10K.ANNOTATION = '/home/jlzheng/dataset/tracking/usot/got10k_flow/train.json'
config.USOT.DATASET.GOT10K.USE = 19000

# LaSOT configs
config.USOT.DATASET.LASOT.PATH = '/home/jlzheng/dataset/tracking/usot/lasot_flow/crop511/'
config.USOT.DATASET.LASOT.ANNOTATION = '/home/jlzheng/dataset/tracking/usot/lasot_flow/train.json'
config.USOT.DATASET.LASOT.USE = 13000

# YT-VOS configs
config.USOT.DATASET.YTVOS.PATH = '/home/jlzheng/dataset/tracking/usot/ytvos_flow/crop511/'
config.USOT.DATASET.YTVOS.ANNOTATION = '/home/jlzheng/dataset/tracking/usot/ytvos_flow/train.json'
config.USOT.DATASET.YTVOS.USE = 4000

# Default training configs for USOT
config.USOT.TRAIN.MODEL = "USOT"
config.USOT.TRAIN.RESUME = False
config.USOT.TRAIN.START_EPOCH = 1
config.USOT.TRAIN.END_EPOCH = 30
config.USOT.TRAIN.TEMPLATE_SIZE = 127
config.USOT.TRAIN.SEARCH_SIZE = 255
config.USOT.TRAIN.MEMORY_NUM = 4
config.USOT.TRAIN.STRIDE = 8
config.USOT.TRAIN.BATCH = 12
config.USOT.TRAIN.BATCH_STAGE_2 = 12
config.USOT.TRAIN.PRETRAIN = 'imagenet_pretrain.model'
config.USOT.TRAIN.MOMENTUM = 0.9
config.USOT.TRAIN.WEIGHT_DECAY = 0.0001
config.USOT.TRAIN.WHICH_USE = ['VID', 'LASOT', 'GOT10K', 'YTVOS']

# Default testing configs for USOT
config.USOT.TEST.MODEL = config.USOT.TRAIN.MODEL
config.USOT.TEST.DATA = 'VOT2018'
config.USOT.TEST.START_EPOCH = 10
config.USOT.TEST.END_EPOCH = 30


def _update_dict(k, v, model_name):
    if k in ['TRAIN', 'TEST']:
        for vk, vv in v.items():
            config[model_name][k][vk] = vv
    elif k == 'DATASET':
        for vk, vv in v.items():
            if vk not in ['VID', 'GOT10K', 'LASOT', 'YTVOS']:
                config[model_name][k][vk] = vv
            else:
                for vvk, vvv in vv.items():
                    config[model_name][k][vk][vvk] = vvv
    else:
        config[k] = v


def update_config(config_file):
    """
    Add new keys to config
    """
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        model_name = list(exp_config.keys())[0]
        if model_name not in ['USOT']:
            raise ValueError('please edit config_usot.py to support new model')

        model_config = exp_config[model_name]
        for k, v in model_config.items():
            if k in config or k in config[model_name]:
                _update_dict(k, v, model_name)   # k = USOT
            else:
                raise ValueError("{} not exist in config_usot.py".format(k))
