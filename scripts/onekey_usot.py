from __future__ import absolute_import
import os
import yaml
import argparse
from os.path import exists

def parse_args():
    """
    args for onekey script.
    """
    parser = argparse.ArgumentParser(description='Train and test USOT with onekey')
    # Config file for train and test
    parser.add_argument('--cfg', type=str, default='experiments/train/USOT.yaml', help='yaml config file name')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Train - test information
    info = yaml.load(open(args.cfg, 'r').read(), Loader=yaml.FullLoader)
    info = info['USOT']
    trainINFO = info['TRAIN']
    testINFO = info['TEST']

    # Epoch training -- train 30 epochs
    if trainINFO['ISTRUE']:
        print('==> train phase')
        print('python ./scripts/train_usot.py --cfg {0} --gpus {1} --workers {2} 2>&1 | tee var/log/usot_train.log'
                  .format(args.cfg, info['GPUS'], info['WORKERS']))

        if not exists('var/log'):
            os.makedirs('var/log')

        os.system('python ./scripts/train_usot.py --cfg {0} --gpus {1} --workers {2} 2>&1 | tee var/log/usot_train.log'
                  .format(args.cfg, info['GPUS'], info['WORKERS']))

    # Epoch testing -- test checkpoints from 10-30 epochs
    if testINFO['ISTRUE']:
        print('==> test phase')
        print('mpiexec -n {0} python ./scripts/test_epochs_usot.py --arch {1} --start_epoch {2} --end_epoch {3} --gpus={4} \
                  --threads {0} --dataset {5} 2>&1 | tee var/log/usot_epoch_test.log'
                  .format(testINFO['THREADS'], trainINFO['MODEL'], testINFO['START_EPOCH'], testINFO['END_EPOCH'],
                          info['GPUS'], testINFO['DATA']))

        if not exists('var/log'):
            os.makedirs('var/log')

        os.system('mpiexec -n {0} python ./scripts/test_epochs_usot.py --arch {1} --start_epoch {2} --end_epoch {3} --gpus={4} \
                  --threads {0} --dataset {5} 2>&1 | tee var/log/epoch_test_usot.log'
                  .format(testINFO['THREADS'], trainINFO['MODEL'], testINFO['START_EPOCH'], testINFO['END_EPOCH'],
                          info['GPUS'], testINFO['DATA']))

        # Eval all checkpoints on the vot benchmarks
        print('====> use new testing toolkit')
        trackers = os.listdir(os.path.join('./var/result', testINFO['DATA']))
        trackers = " ".join(trackers)
        if 'VOT' in testINFO['DATA']:
            print('python lib/eval_toolkit/bin/eval.py --dataset_dir datasets_test --dataset {0}\
              --tracker_result_dir var/result/{0} --trackers {1}'.format(testINFO['DATA'], trackers))
            os.system('python lib/eval_toolkit/bin/eval.py --dataset_dir datasets_test --dataset {0}\
             --tracker_result_dir var/result/{0} --trackers {1} 2>&1 | tee var/log/eval_epochs_usot.log'.format(testINFO['DATA'], trackers))
        else:
            raise ValueError('Your wanted dataset not supported now, please add new dataset')


if __name__ == '__main__':
    main()
