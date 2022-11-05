''' test the pretrain autoencoder

author:
    zhangsihao yang

logs:
    20221101    file created
'''
import argparse

import torch
from dataset import Dataset
from model import Network
from options import options, update_options
from single_gpu_train import _fix_random, parse_args


def test():
    # visit the dataset and get the features

    # partation the features 10 times to get the accuracy

    pass


if __name__ is '__main__':
    parse_args()

    # set up the dataset and dataloader
    _fix_random(options.seed)

    # make the dataset and dataloader
    train_op = options.data.train
    if train_op is not None:
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

    # set up the network
    net = Network(**options.model.params).cuda()
    # load the pretrained weights
    net.load_state_dict(torch.load(f'/runtime/300.pt')['net'])
    net.eval()

    # do the test
    test()
