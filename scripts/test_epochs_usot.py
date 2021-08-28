import os
import time
import argparse
from mpi4py import MPI


parser = argparse.ArgumentParser(description='multi-gpu test all epochs')
parser.add_argument('--arch', dest='arch', default='USOT',
                    help='architecture of model')
parser.add_argument('--start_epoch', default=10, type=int, required=True, help='test start epoch')
parser.add_argument('--end_epoch', default=30, type=int, required=True,
                    help='test end epoch')
parser.add_argument('--gpus', default="0,1,2,3", type=str, required=True, help='gpus')
parser.add_argument('--threads', default=11, type=int, required=True)
parser.add_argument('--dataset', default='VOT2018', type=str, help='benchmark to test')
args = parser.parse_args()

# Init gpu and epochs
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
gpu_list = args.gpus.split(",")
gpu_nums = len(gpu_list)
GPU_ID = int(gpu_list[0]) + rank % gpu_nums

# Get the name of the node
node_name = MPI.Get_processor_name()
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
print("node name: {}, GPU_ID: {}".format(node_name, GPU_ID))
time.sleep(rank * 5)
num_total_tests = args.end_epoch - args.start_epoch + 1
num_test_each_thread = num_total_tests // args.threads + 1

# Run test scripts -- two epochs for each thread
for i in range(num_test_each_thread):
    arch = args.arch
    dataset = args.dataset
    try:
        epoch_ID += args.threads
    except:
        epoch_ID = rank % (args.end_epoch - args.start_epoch + 1) + args.start_epoch

    if epoch_ID > args.end_epoch:
        continue

    resume = 'var/snapshot/checkpoint_e{}.pth'.format(epoch_ID)
    print('==> test {}th epoch'.format(epoch_ID))
    os.system('python ./scripts/test_usot.py --arch {0} --resume {1} --dataset {2} --epoch_test True'
              .format(arch, resume, dataset))
