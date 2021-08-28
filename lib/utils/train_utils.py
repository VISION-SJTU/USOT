import math
import time, os, torch, logging
from os.path import join, exists
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


def is_valid_number(x):
    return not(math.isnan(x) or math.isinf(x) or x > 1e4)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def create_logger(cfg, modelFlag='USOT', phase='train'):
    root_output_dir = cfg.OUTPUT_DIR
    # Set up logger
    if not exists(root_output_dir):
        print('=> creating {}'.format(root_output_dir))
        os.makedirs(root_output_dir)
    cfg = cfg[modelFlag]
    model = cfg.TRAIN.MODEL

    final_output_dir = join(root_output_dir, model)

    print('=> creating {}'.format(final_output_dir))
    os.makedirs(final_output_dir, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(model, time_str, phase)
    final_log_file = join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = join(root_output_dir, model, (model + '_' + time_str))
    print('=> creating {}'.format(tensorboard_log_dir))
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def print_speed(i, i_time, n, logger):

    average_time = i_time
    remaining_time = (n - i) * average_time
    remaining_day = math.floor(remaining_time / 86400)
    remaining_hour = math.floor(remaining_time / 3600 - remaining_day * 24)
    remaining_min = math.floor(remaining_time / 60 - remaining_day * 1440 - remaining_hour * 60)
    logger.info('Progress: %d / %d [%d%%], Speed: %.3f s/iter, ETA %d:%02d:%02d (D:H:M)\n'
                % (i, n, i/n*100, average_time, remaining_day, remaining_hour, remaining_min))
    logger.info('\nPROGRESS: {:.2f}%\n'.format(100 * i / n))


def save_model(model, epoch, optimizer, model_name, cfg, isbest=False):
    """
    Saving model
    """
    if not exists(cfg.CHECKPOINT_DIR):
        os.makedirs(cfg.CHECKPOINT_DIR)

    if epoch >= 5:
        save_checkpoint({
            'epoch': epoch,
            'arch': model_name,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict()
        }, isbest, cfg.CHECKPOINT_DIR, 'checkpoint_e%d.pth' % (epoch))
    else:
        print('Checkpoint for this epoch does not save (<5).')


def load_pretrain(model, pretrained_path, print_unuse=True, gpus=None):
    print('load pretrained model from {}'.format(pretrained_path))

    if gpus is not None:
        torch.cuda.set_device(gpus[0])
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
        pretrained_dict = remove_prefix(pretrained_dict, 'feature_extractor.')  # remove online train
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')  # remove multi-gpu label
        pretrained_dict = remove_prefix(pretrained_dict, 'feature_extractor.')   # remove online train

    # Since we adopt a slightly different ResNet
    # We have to convert the model loading logic a little bit when loading moco
    if "moco" in pretrained_path:
        processed_dict = dict()
        ori_keys = list(pretrained_dict.keys())
        for ori_key in ori_keys:
            if "encoder_q" in ori_key:
                drop_list = ["encoder_q.layer2.0.downsample.0.weight", "encoder_q.layer3.0.downsample.0.weight"]
                revised_key = ori_key.replace("encoder_q", "features.features")
                if ori_key in drop_list:
                    size = pretrained_dict[ori_key].shape
                    correlation_core_33 = torch.Tensor(torch.zeros([size[0], size[1], 3, 3])).cuda()
                    correlation_core_33[:, :, 1, 1] = pretrained_dict[ori_key][:, :, 0, 0]
                    processed_dict[revised_key] = correlation_core_33
                else:
                    processed_dict[revised_key] = pretrained_dict[ori_key]
        check_keys(model, processed_dict, print_unuse=print_unuse)
        model.load_state_dict(processed_dict, strict=False)
    else:
        check_keys(model, pretrained_dict, print_unuse=print_unuse)
        model.load_state_dict(pretrained_dict, strict=False)
    return model


def remove_prefix(state_dict, prefix):
    """
    Old style model is stored with all names of parameters share common prefix 'module.'
    """
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def check_keys(model, pretrained_state_dict, print_unuse=True):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = list(ckpt_keys - model_keys)
    missing_keys = list(model_keys - ckpt_keys)

    # Remove num_batches_tracked
    for k in sorted(missing_keys):
        if 'num_batches_tracked' in k:
            missing_keys.remove(k)

    print('missing keys:{}'.format(missing_keys))
    if print_unuse:
        print('unused checkpoint keys:{}'.format(unused_pretrained_keys))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth.tar'):
    """
    Save checkpoint
    """
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def restore_from(model, optimizer, ckpt_path):
    print('restore from {}'.format(ckpt_path))
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path, map_location = lambda storage, loc: storage.cuda(device))
    epoch = ckpt['epoch']
    arch = ckpt['arch']
    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, epoch,  arch


# ----------------------------
# Build lr (from SiamRPN++)
# ---------------------------
class LRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        if 'lr_spaces' not in self.__dict__:
            raise Exception('lr_spaces must be set in "LRSchduler"')
        super(LRScheduler, self).__init__(optimizer, last_epoch)

    def get_cur_lr(self):
        return self.lr_spaces[self.last_epoch]

    def get_lr(self):
        epoch = self.last_epoch
        return [self.lr_spaces[epoch] * pg['initial_lr'] / self.start_lr
                for pg in self.optimizer.param_groups]

    def __repr__(self):
        return "({}) lr spaces: \n{}".format(self.__class__.__name__,
                                             self.lr_spaces)


class LogScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.03, end_lr=5e-4,
                 epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.epochs = epochs
        self.lr_spaces = np.logspace(math.log10(start_lr),
                                     math.log10(end_lr),
                                     epochs)

        super(LogScheduler, self).__init__(optimizer, last_epoch)


class StepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=None,
                 step=10, mult=0.1, epochs=50, last_epoch=-1, **kwargs):
        if end_lr is not None:
            if start_lr is None:
                start_lr = end_lr / (mult ** (epochs // step))
            else:  # for warm up policy
                mult = math.pow(end_lr/start_lr, 1. / (epochs // step))
        self.start_lr = start_lr
        self.lr_spaces = self.start_lr * (mult**(np.arange(epochs) // step))
        self.mult = mult
        self._step = step

        super(StepScheduler, self).__init__(optimizer, last_epoch)


class MultiStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=None,
                 steps=[10, 20, 30, 40], mult=0.5, epochs=50,
                 last_epoch=-1, **kwargs):
        if end_lr is not None:
            if start_lr is None:
                start_lr = end_lr / (mult ** (len(steps)))
            else:
                mult = math.pow(end_lr/start_lr, 1. / len(steps))
        self.start_lr = start_lr
        self.lr_spaces = self._build_lr(start_lr, steps, mult, epochs)
        self.mult = mult
        self.steps = steps

        super(MultiStepScheduler, self).__init__(optimizer, last_epoch)

    def _build_lr(self, start_lr, steps, mult, epochs):
        lr = [0] * epochs
        lr[0] = start_lr
        for i in range(1, epochs):
            lr[i] = lr[i-1]
            if i in steps:
                lr[i] *= mult
        return np.array(lr, dtype=np.float32)


class LinearStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=0.005,
                 epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_spaces = np.linspace(start_lr, end_lr, epochs)
        super(LinearStepScheduler, self).__init__(optimizer, last_epoch)


class CosStepScheduler(LRScheduler):
    def __init__(self, optimizer, start_lr=0.01, end_lr=0.005,
                 epochs=50, last_epoch=-1, **kwargs):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_spaces = self._build_lr(start_lr, end_lr, epochs)

        super(CosStepScheduler, self).__init__(optimizer, last_epoch)

    def _build_lr(self, start_lr, end_lr, epochs):
        index = np.arange(epochs).astype(np.float32)
        lr = end_lr + (start_lr - end_lr) * \
            (1. + np.cos(index * np.pi / epochs)) * 0.5
        return lr.astype(np.float32)


class WarmUPScheduler(LRScheduler):
    def __init__(self, optimizer, warmup, normal, epochs=50, last_epoch=-1):
        warmup = warmup.lr_spaces  # [::-1]
        normal = normal.lr_spaces
        self.lr_spaces = np.concatenate([warmup, normal])
        self.start_lr = normal[0]

        super(WarmUPScheduler, self).__init__(optimizer, last_epoch)


LRs = {
    'log': LogScheduler,
    'step': StepScheduler,
    'multi-step': MultiStepScheduler,
    'linear': LinearStepScheduler,
    'cos': CosStepScheduler}


def _build_lr_scheduler(optimizer, config, epochs=50, last_epoch=-1):
    return LRs[config.TYPE](optimizer, last_epoch=last_epoch,
                            epochs=epochs, **config.KWARGS)


def _build_warm_up_scheduler(optimizer, cfg, epochs=50, last_epoch=-1):

    warmup_epoch = cfg.TRAIN.WARMUP.EPOCH
    sc1 = _build_lr_scheduler(optimizer, cfg.TRAIN.WARMUP,
                              warmup_epoch, last_epoch)
    sc2 = _build_lr_scheduler(optimizer, cfg.TRAIN.LR,
                              epochs - warmup_epoch, last_epoch)
    return WarmUPScheduler(optimizer, sc1, sc2, epochs, last_epoch)


def build_lr_scheduler(optimizer, cfg, epochs=50, last_epoch=-1, model_FLAG='USOT'):
    cfg = cfg[model_FLAG]
    if cfg.TRAIN.WARMUP.IFNOT:
        return _build_warm_up_scheduler(optimizer, cfg, epochs, last_epoch)
    else:
        return _build_lr_scheduler(optimizer, cfg.TRAIN.LR, epochs, last_epoch)
