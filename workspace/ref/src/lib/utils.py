''' Utilities about GPU usage.
Modified from: https://github.com/facebookresearch/dino/blob/main/utils.py
Author: Zhangsihao Yang
Date: 2022-0314
'''
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist


def _fix_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for reproducibility
    torch.backends.cudnn.deterministic = True
    # for speed
    # since we are using minkowski engine, the network could
    # not be speeded up with this option in my understanding
    torch.backends.cudnn.benchmark = False

def _set_local_rank(args):
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])


def _set_local_single(args):
    print('Will run the code on one GPU.')
    args.rank, args.gpu, args.world_size = 0, 0, 1
    os.environ['MASTER_ADDR'] = '127.0.0.0'
    os.environ['MASTER_PORT'] = '29500'


def _set_slurm_rank(args):
    args.rank = int(os.environ['SLURM_PROCID'])
    args.gpu = args.rank % torch.cuda.device_count()


def _set_rank(args):
    if ('RANK' in os.environ) and ('WORLD_SIZE' in os.environ):
        _set_local_rank(args)
    elif 'SLURM_PROCID' in os.environ:
        _set_slurm_rank()
    elif torch.cuda.is_available():
        _set_local_single(args)
    else:
        sys.exit('Does not support training without GPU!')


def _setup_for_distributed(is_master: bool):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print


def init_distributed_mode(args):
    _fix_random(args.seed)

    _set_rank(args)
    dist.init_process_group(
        backend='nccl', 
        world_size=args.world_size,
        rank=args.rank
    )

    torch.cuda.set_device(args.gpu)

    print(f'| distributed init (rank {args.rank}): ')

    dist.barrier()

    _setup_for_distributed(args.rank==0)


def _is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def _get_rank():
    if not _is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return _get_rank() == 0
