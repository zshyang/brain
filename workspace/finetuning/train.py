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


def create_label_dict():
    return {
        'AD_pos': 0,
        'MCI_neg': 1,
        'MCI_pos': 2,
        'NL_neg': 3,
        'NL_pos': 4
    }


def load_dataloader(opt):
    dataset = Dataset(**opt.dataset.params)
    collate_fn = getattr(
        dataset, str(opt.dataloader.collate_fn), None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        collate_fn=collate_fn,
        batch_size=opt.dataloader.batch_size,
        drop_last=bool(opt.dataloader.drop_last),
        num_workers=int(opt.dataloader.num_workers)
    )
    return dataloader


def finetune():
    pass


if __name__ == '__main__':
    parse_args()

    _fix_random(options.seed)

    #======== dataloader ========
    train_op = options.data.train
    train_dataloader = load_dataloader(train_op)

    val_op = options.data.val
    val_dataloader = load_dataloader(val_op)

    test_op = options.data.test
    test_dataloader = load_dataloader(test_op)

    #======== net, optimizer, scheduler ========
    net = Network(**options.model.params).cuda()

    #======== finetune ========
    finetune()
