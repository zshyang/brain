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
    # dataset = 
    # self.opt = opt

    # import lib.datasets as dataset
    # dataset_lib = self.opt.dataset.lib
    # self.dataset = getattr(
    #     dataset, dataset_lib
    # )(
    #     **self.opt.dataset.params
    # )
    train_dataset = Dataset(**options.data.train.dataset.params)

    self.collate_fn = getattr(
        collateFunctions, 
        str(self.opt.dataloader.collate_fn),
        None
    )
    if self.collate_fn is None:
        self.collate_fn = getattr(self.dataset, str(self.opt.dataloader.collate_fn), None)

    self.sampler = torch.utils.data.DistributedSampler(
        self.dataset, shuffle=True
    )

    self.dataloader = torch.utils.data.DataLoader(
        self.dataset, 
        sampler=self.sampler,
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
