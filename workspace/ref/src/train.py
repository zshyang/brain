''' train script.

name convention:
    tvtdp = train validation test data provider

class

functions:
    parse_args()


author:
    Zhangsihao Yang

date: 
    2022-0314

logs:
    
'''
from __future__ import print_function

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from lib.data_provider import TVTDistributedDataProvider
# from lib.dataset import ModelNetDataset, ShapeNetDataset
from lib.model_logger import ModelLogger
from lib.model_manager import ModelManager
from lib.options import options, update_options
from lib.utils import init_distributed_mode, is_main_process


def parse_args():
    parser = argparse.ArgumentParser()

    str_help = 'experiment options file name'
    parser.add_argument(
        '--options',
        help=str_help, required=True, type=str
    )

    str_help = 'required by distributed training'
    parser.add_argument(
        '--local_rank',
        help=str_help, type=int
    )

    args = parser.parse_args()

    update_options(args.options)


if __name__  == '__main__':
    parse_args()

    init_distributed_mode(options)

    logger = ModelLogger(options, True)

    tvtdp = TVTDistributedDataProvider(options.data)

    import lib.models as model
    net = getattr(
        model, options.model.lib
    )(**options.model.params).cuda()
    net = nn.parallel.DistributedDataParallel(
        net, device_ids=[options.gpu],
        find_unused_parameters=False
    )

    optimizer = getattr(
        optim, options.optim.name
    )(
        net.parameters(),
        **options.optim.params
    )
    scheduler = getattr(
        optim.lr_scheduler,
        options.optim.scheduler.name,
        None
    )
    if scheduler is not None:
        scheduler = scheduler(
            optimizer, **options.optim.scheduler.params
        )

    model_manager = ModelManager(
        options.outf, options.manager, True
    )

    import lib.trainers as trainer
    model_trainer = getattr(
        trainer, options.train.lib
    )(
        logger, tvtdp, net, optimizer, scheduler,
        model_manager
    )
    model_trainer.train(options)
