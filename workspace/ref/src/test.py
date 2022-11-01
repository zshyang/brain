''' test script.

author
    Zhangsihao Yang

logs:
    20220320
        file created
    20220918
        update test import tester from name to lib
'''
from __future__ import print_function

import argparse
import os
import random
import sys

# import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

# import lib.trainer as trainer
from lib.data_provider import TVTDistributedDataProvider
# from lib.dataset import ModelNetDataset, ShapeNetDataset
from lib.model_logger import ModelLogger
from lib.model_manager import ModelManager
from lib.options import options, update_options
from lib.utils import init_distributed_mode, is_main_process


def parse_args():
    parser = argparse.ArgumentParser()

    str_help = 'experiment options file name'
    parser.add_argument('--options',
        help=str_help, required=True, type=str)

    str_help = 'required by distributed training'
    parser.add_argument('--local_rank',
        help=str_help, type=int)

    args = parser.parse_args()

    update_options(args.options)


if __name__  == '__main__':
    parse_args()

    init_distributed_mode(options)

    logger = ModelLogger(options, True)

    # train_data_provider = DistributedDataProvider(
    #     options.data.train)
    # test_data_provider = DistributedDataProvider(
    #     options.data.test)
    tvtdp = TVTDistributedDataProvider(options.data)
    
    # import model after options are updated
    import lib.models as model
    net = getattr(
        model, options.model.lib
    )(
        **options.model.params
    ).cuda()
    net = nn.parallel.DistributedDataParallel(
        net, device_ids=[options.gpu],
        find_unused_parameters=False
    )

    model_manager = ModelManager(
        options.outf, options.manager, True
    )

    import lib.testers as tester
    model_tester = getattr(
        tester, 
        options.test.lib # update
    )(
        logger, tvtdp, net,
        model_manager
    )
    model_tester.test(options)
