''' fine tuning on the pre-trained model.

author: 
    zhangsihao yang

logs:
    20221106 file created
'''
import argparse
import random
import time

import numpy as np
import pytorch3d.loss
import torch
import torch.optim as optim

from dataset import Dataset
from model import Network
from options import options, update_options


def parse_args():
    parser = argparse.ArgumentParser()

    str_help = 'experiment options file name'
    parser.add_argument(
        '--options',
        help=str_help, required=True, type=str
    )

    args = parser.parse_args()

    update_options(args.options)


def _fix_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_net_from_pre_trained(net, ckpt_path):
    return net


if __name__ == '__main__':
    parse_args()

    _fix_random(options.seed)

    train_op = options.data.train
    train_dataset = Dataset(**train_op.dataset.params)
    collate_fn = getattr(
        train_dataset, str(train_op.dataloader.collate_fn), None
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        collate_fn=collate_fn,
        batch_size=train_op.dataloader.batch_size,
        drop_last=bool(train_op.dataloader.drop_last),
        num_workers=int(train_op.dataloader.num_workers)
    )

