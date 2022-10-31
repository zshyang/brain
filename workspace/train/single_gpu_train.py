''' train the model with single gpu.

author:
    zhangsihao yang

logs:
    20221030    file created
'''
import argparse
import random

import numpy as np
import torch
from dataset import Dataset

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


if __name__ == '__main__':
    parse_args()

    _fix_random(options.seed)

    # make the dataset
    train_dataset = Dataset(**options.data.train.dataset.params)

    collate_fn = getattr(train_dataset, str(options.data.train.dataloader.collate_fn), None)

    train_dataloader = torch.utils.data.DataLoader(
        self.dataset, 
        collate_fn=self.collate_fn,
        batch_size=self.opt.dataloader.batch_size,
        drop_last=bool(self.opt.dataloader.drop_last),
        num_workers=int(self.opt.dataloader.num_workers)
    )
    self.train_iter = iter(self.dataloader)

    self.epoch = 0

    # make the dataload

    # make model

    # make optimizer and scheduler

    # train the network
