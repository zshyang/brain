''' trian the network

author:
    zhangsihao yang

logs:
    20230221: file created
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


def train():
    for i in range(options.train.max_epoch + 1):
        for in_data, target in train_dataloader:
            print(in_data)
            print(target)
            break


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
        num_workers=int(train_op.dataloader.num_workers),
        shuffle=bool(train_op.dataloader.shuffle)
    )

    net = Network(**options.model.params).cuda()

    train()
