''' trian the network

author:
    zhangsihao yang

logs:
    20230221: file created
'''
import argparse
import random
import time

import model
import numpy as np
import pytorch3d.loss
import torch
import torch.optim as optim
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


def _dict_cuda(dict_tensor):
    ''' this function is to move 
    a dict of tensors onto gpu
    '''
    for key in dict_tensor:
        if type(dict_tensor[key]) is list:
            continue
        if type(dict_tensor[key]) is dict:
            self._dict_cuda(dict_tensor[key])
        if type(dict_tensor[key]) is torch.Tensor:
            dict_tensor[key] = dict_tensor[key].cuda()


def _cuda(in_data, target):
    ''' move input data and 
    target onto gpu
    '''
    _dict_cuda(in_data)
    _dict_cuda(target)


def _criterion(pred, y, is_vae=False, kl_weight=0., **kwargs):
    ''' Compute the reconstruction loss and KLD if 
    needed.
    Args:
        y is the output point cloud (B, N, 3)
    '''

    if is_vae:
        y_, mean, log_var = pred
    else:
        y_, _ = pred

    # compute chamfer distance
    loss, _ = pytorch3d.loss.chamfer_distance(x=y_, y=y)
    return loss


def train():
    for i in range(options.train.max_epoch + 1):
        for in_data, target in train_dataloader:
            net.train()
            _cuda(in_data, target)
            pred = net(**in_data)
            loss = _criterion(pred, **target)
            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
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

    net = getattr(model, options.model.lib)(**options.model.params).cuda()
    # Network(**options.model.params).cuda()

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

    train()
