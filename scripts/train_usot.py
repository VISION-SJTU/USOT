# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Zhipeng Zhang (zhangzhipeng2017@ia.ac.cn)
# Revised by Jilai Zheng, for USOT
# ------------------------------------------------------------------------------

import os
import shutil
import time
import math
import pprint
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch.nn as nn

import torch
from torch.utils.data import DataLoader

import lib.models.models as models
from lib.utils.train_utils import is_valid_number, AverageMeter, \
    create_logger, print_speed, load_pretrain, restore_from, save_model, build_lr_scheduler
from lib.dataset_loader.datasets_usot import USOTDataset
from lib.config.config_usot import config, update_config


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Train USOT')
    # File for basic configs
    parser.add_argument('--cfg', type=str,
                        default='experiments/train/USOT.yaml', help='yaml configure file name')
    parser.add_argument('--gpus', type=str, help='gpus')
    parser.add_argument('--workers', type=int, help='num of dataloader workers')

    args, rest = parser.parse_known_args()
    # Update config with config file
    update_config(args.cfg)
    args = parser.parse_args()

    return args


def reset_config(config, args):
    """
    set gpus and workers
    """
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers


def check_trainable(model, logger):
    """
    print trainable params info
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info('trainable params:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(name)

    assert len(trainable_params) > 0, 'no trainable parameters'

    return trainable_params


def build_opt_lr(cfg, model, current_epoch=1):
    # Fix all backbone layers at first
    for param in model.features.features.parameters():
        param.requires_grad = False
    for m in model.features.features.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    if current_epoch >= cfg.USOT.TRAIN.UNFIX_EPOCH:

        if len(cfg.USOT.TRAIN.TRAINABLE_LAYER) > 0:
            # Train specific layers in backbone
            for layer in cfg.USOT.TRAIN.TRAINABLE_LAYER:
                for param in getattr(model.features.features, layer).parameters():
                    param.requires_grad = True
                for m in getattr(model.features.features, layer).modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.train()
        else:
            # Train all backbone layers
            for param in model.features.features.parameters():
                param.requires_grad = True
            for m in model.features.features.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
    else:
        for param in model.features.features.parameters():
            param.requires_grad = False
        for m in model.features.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    trainable_params = []
    if current_epoch >= cfg.USOT.TRAIN.UNFIX_EPOCH:
        trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                               model.features.features.parameters()),
                              'lr': cfg.USOT.TRAIN.LAYERS_LR * cfg.USOT.TRAIN.BASE_LR}]
    else:
        trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                               model.features.features.parameters()),
                              'lr': 0.1 * cfg.USOT.TRAIN.BASE_LR}]

    try:
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': cfg.USOT.TRAIN.BASE_LR}]
    except:
        pass

    trainable_params += [{'params': model.connect_model.parameters(),
                          'lr': cfg.USOT.TRAIN.BASE_LR}]

    # Print trainable parameter (first check)
    print('==========first check trainable==========')
    for param in trainable_params:
        print(param)

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.USOT.TRAIN.MOMENTUM,
                                weight_decay=cfg.USOT.TRAIN.WEIGHT_DECAY)

    lr_scheduler = build_lr_scheduler(optimizer, cfg, epochs=cfg.USOT.TRAIN.END_EPOCH)
    lr_scheduler.step(cfg.USOT.TRAIN.START_EPOCH - 1)

    return optimizer, lr_scheduler


def usot_train(train_loader, model, optimizer, epoch,
               cur_lr, cfg, writer_dict, logger, device):
    # Prepare average meter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses_memory = AverageMeter()
    cls_losses_ori = AverageMeter()
    reg_losses = AverageMeter()
    end = time.time()

    # Switching the model to train mode
    # We find that our 4*3090 machine has a bad power supply.
    # So sleeping for a while after sending the model to GPU is necessary for preventing reboot.
    print("After sending model to device, sleep 5s for preventing reboot.")
    model.train()
    model = model.to(device)
    time.sleep(5)
    print("Sleeping finished.")

    for iter, input in enumerate(train_loader):

        # Measure data loading time
        data_time.update(time.time() - end)

        # Input and output/loss
        label_cls = input[2].type(torch.FloatTensor)  # BCE need float
        template = input[0].to(device)
        search = input[1].to(device)
        label_cls = label_cls.to(device)
        reg_label = input[3].float().to(device)
        reg_weight = input[4].float().to(device)
        template_bbox = input[5].float().to(device)

        if len(input) >= 7:
            # Cycle memory training
            search_memory = input[6].to(device)
            search_bbox = input[7].to(device)
        else:
            search_memory = None
            search_bbox = None

        # The following codes determines the linear weights for forward tracking in training
        # The response map of forward tracking is a linear combination of online and offline tracking module
        # Namely, forward_res_map = cls_ratio * off_forward_cls + (1-cls_ratio) * mem_forward_cls
        # Since the online tracker will become more reliable gradually, cls_ratio decreases with epoch number
        cls_ratio_shift_epochs = cfg.USOT.TRAIN.CLS_RATIO_SHIFT_EPOCHS
        cls_ratios_list = cfg.USOT.TRAIN.CLS_RATIOS
        cls_ratio = None
        for i_ep in range(len(cls_ratio_shift_epochs) - 1):
            if cls_ratios_list[i_ep] <= epoch <= cls_ratios_list[i_ep + 1]:
                cls_ratio = cls_ratios_list[i_ep]
                break
        if cls_ratio is None:
            cls_ratio = cls_ratios_list[-1]

        # Model forward logic
        cls_loss_ori, cls_loss_memory, reg_loss = model(template, search, label_cls,
                                                        reg_target=reg_label, reg_weight=reg_weight,
                                                        template_bbox=template_bbox, search_memory=search_memory,
                                                        search_bbox=search_bbox, cls_ratio=cls_ratio)
        # Offline cls loss and bbox regression loss
        cls_loss_ori = torch.mean(cls_loss_ori)
        reg_loss = torch.mean(reg_loss)

        if cls_loss_memory is not None:
            # With cycle memory
            cls_loss_memory = torch.mean(cls_loss_memory)
            done = False
            # The following codes determines the linear weights for loss function
            # You can set a dynamic lambda setting by tuning the following configs
            lambda_shift_epochs = cfg.USOT.TRAIN.LAMBDA_SHIFT_EPOCHS
            lambda1_list = cfg.USOT.TRAIN.LAMBDA_1_LIST
            lambda_total = cfg.USOT.TRAIN.LAMBDA_TOTAL
            for i_ep in range(len(lambda_shift_epochs) - 1):
                if lambda_shift_epochs[i_ep] <= epoch <= lambda_shift_epochs[i_ep + 1]:
                    loss = lambda1_list[i_ep] * cls_loss_ori + \
                           (lambda_total - lambda1_list[i_ep]) * cls_loss_memory + 1.0 * reg_loss
                    done = True
                    break
            if not done:
                # From last epoch in lambda1_list to the final epoch
                lambda_1 = lambda1_list[-1]
                loss = lambda_1 * cls_loss_ori + (lambda_total - lambda_1) * cls_loss_memory + 1.0 * reg_loss
        else:
            # Without cycle memory
            cls_loss_memory = 0
            # Loss for naive Siamese training
            loss = cfg.USOT.TRAIN.LAMBDA_1_NAIVE * cls_loss_ori + 1.0 * reg_loss

        loss = torch.mean(loss)

        # Compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()

        if is_valid_number(loss.item()):
            optimizer.step()

        # Record loss
        loss = loss.item()
        losses.update(loss, template.size(0))

        cls_loss_ori = cls_loss_ori.item()
        cls_losses_ori.update(cls_loss_ori, template.size(0))

        try:
            cls_loss_memory = cls_loss_memory.item()
        except:
            cls_loss_memory = 0

        cls_losses_memory.update(cls_loss_memory, template.size(0))

        reg_loss = reg_loss.item()
        reg_losses.update(reg_loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.PRINT_FREQ == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}] lr: {lr:.7f}\t Batch Time: {batch_time.avg:.3f}s \t Data Time:{data_time.avg:.3f}s \t CLS_ORI Loss:{cls_loss_ori.avg:.5f} \t CLS_MEMORY Loss:{cls_loss_memory.avg:.5f} \t REG Loss:{reg_loss.avg:.5f} \t Loss:{loss.avg:.5f}'.format(
                    epoch, iter + 1, len(train_loader), lr=cur_lr, batch_time=batch_time, data_time=data_time,
                    loss=losses, cls_loss_ori=cls_losses_ori, cls_loss_memory=cls_losses_memory, reg_loss=reg_losses))

            print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.USOT.TRAIN.END_EPOCH * len(train_loader), logger)

        # Write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar('train_loss', loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict


def main():
    # Init args, loggers and tensorboard
    args = parse_args()
    reset_config(config, args)

    logger, _, tb_log_dir = create_logger(config, 'USOT', 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
    }

    if not os.path.exists('./pretrain'):
        os.makedirs('./pretrain')

    model_settings = {'mem_size': config.USOT.TRAIN.MEMORY_NUM, 'pr_pool': True}
    model = models.__dict__[config.USOT.TRAIN.MODEL](model_settings).cuda()  # Build model
    print(model)

    model = load_pretrain(model, './pretrain/{0}'.format(config.USOT.TRAIN.PRETRAIN))  # Load pretrain

    # Get optimizer
    if not config.USOT.TRAIN.START_EPOCH == config.USOT.TRAIN.UNFIX_EPOCH:
        optimizer, lr_scheduler = build_opt_lr(config, model, config.USOT.TRAIN.START_EPOCH)
    else:
        optimizer, lr_scheduler = build_opt_lr(config, model, 1)  # Resume wrong (last line)

    # Check trainable again
    print('==========double check trainable==========')
    trainable_params = check_trainable(model, logger)  # Print trainable params info

    if config.USOT.TRAIN.RESUME and config.USOT.TRAIN.START_EPOCH != 1:  # Resume from checkpoint
        model, optimizer, args.start_epoch, arch = restore_from(model, optimizer, config.USOT.TRAIN.RESUME)

    # Parallel GPU training
    gpus = [int(i) for i in config.GPUS.split(',')]
    gpu_num = len(gpus)
    logger.info('GPU NUM: {:2d}'.format(len(gpus)))

    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model, device_ids=gpus).to(device)

    logger.info(lr_scheduler)
    logger.info('model prepare done')

    # Mian training loop
    for epoch in range(config.USOT.TRAIN.START_EPOCH, config.USOT.TRAIN.END_EPOCH + 1):

        # Build USOT dataloader
        if config.USOT.TRAIN.MODEL == 'USOT':
            train_set = USOTDataset(config)
            train_set.cycle_memory = False
        else:
            assert False, "Warning: Model should be USOT, but currently {}.".format(
                config.USOT.TRAIN.MODEL)

        if epoch >= config.USOT.TRAIN.MEMORY_EPOCH:
            # Training with cycle memory scheme
            train_set.cycle_memory = True
            train_loader = DataLoader(train_set, batch_size=config.USOT.TRAIN.BATCH_STAGE_2 * gpu_num,
                                      num_workers=config.WORKERS,
                                      pin_memory=True, sampler=None, drop_last=True)
        else:
            # Training naive Siamese tracker
            train_loader = DataLoader(train_set, batch_size=config.USOT.TRAIN.BATCH * gpu_num,
                                      num_workers=config.WORKERS,
                                      pin_memory=True, sampler=None, drop_last=True)

        # Check if it's time to unfix and train the backbone
        if epoch == config.USOT.TRAIN.UNFIX_EPOCH:
            logger.info('training backbone')
            optimizer, lr_scheduler = build_opt_lr(config, model.module, epoch)
            print('==========double check trainable==========')
            check_trainable(model, logger)  # Print trainable params info

        lr_scheduler.step(epoch - 1)
        curLR = lr_scheduler.get_cur_lr()

        # The main training logic for usot
        model, writer_dict = usot_train(train_loader, model, optimizer, epoch,
                                        curLR, config, writer_dict, logger, device=device)

        # Save model
        save_model(model, epoch, optimizer, config.USOT.TRAIN.MODEL, config, isbest=False)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
