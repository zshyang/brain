''' trian the network

author:
    zhangsihao yang

logs:
    20230221: file created
'''
import argparse
import os
import random
import time

import model
import numpy as np
import pytorch3d.loss
import torch
import torch.nn.functional as F
import torch.optim as optim
from dataset import Dataset
from early_stop import EarlyStopping, Stop_args
from options import options, update_options
from writer import Writer


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


def _criterion(pred, y, **kwargs):
    ''' Compute the reconstruction loss and KLD if 
    needed.
    Args:
        y is the output point cloud (B, N, 3)
    '''

    loss = F.cross_entropy(pred, y)

    num_correct = torch.sum(torch.argmax(pred, dim=1) == y).item()
    num_pass = y.shape[0]

    return loss, [num_correct, num_pass]


def train():
    if train_dataloader is None:
        return

    for in_data, target in train_dataloader:
        net.train()
        _cuda(in_data, target)
        pred = net(**in_data)

        loss, info = _criterion(pred, **target)
        if torch.isnan(loss):
            print('loss is nan')
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

        # record
        writer.add_scalar('train_loss', loss.item(), epoch)
        writer.add_scalar('train_num_correct', info[0], epoch)
        writer.add_scalar('train_num_pass', info[1], epoch)


def validate():
    if val_dataloader is None:
        return

    with torch.no_grad():
        net.eval()
        for in_data, target in val_dataloader:
            _cuda(in_data, target)
            pred = net(**in_data)

            loss, info = _criterion(pred, **target)

            # record
            writer.add_scalar('val_loss', loss.item(), epoch)
            writer.add_scalar('val_num_correct', info[0], epoch)
            writer.add_scalar('val_num_pass', info[1], epoch)


def test():
    if test_dataloader is None:
        return

    with torch.no_grad():
        net.eval()
        for in_data, target in test_dataloader:
            _cuda(in_data, target)
            pred = net(**in_data)

            loss, info = _criterion(pred, **target)

            # record
            writer.add_scalar('test_loss', loss.item(), epoch)
            writer.add_scalar('test_num_correct', info[0], epoch)
            writer.add_scalar('test_num_pass', info[1], epoch)


def create_dataloader(data_op):
    if data_op is None:
        return None

    dataset = Dataset(**data_op.dataset.params)
    collate_fn = getattr(
        dataset, str(data_op.dataloader.collate_fn), None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=data_op.dataloader.batch_size,
        drop_last=bool(data_op.dataloader.drop_last),
        num_workers=int(data_op.dataloader.num_workers),
        shuffle=bool(data_op.dataloader.shuffle)
    )
    return dataloader


if __name__ == '__main__':
    parse_args()

    _fix_random(options.seed)

    # create dataloaders
    train_dataloader = create_dataloader(options.data.train)
    val_dataloader = create_dataloader(options.data.val)
    test_dataloader = create_dataloader(options.data.test)

    net = getattr(model, options.model.lib)(**options.model.params).cuda()

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

    writer = Writer(
        'log.txt', os.path.join('/workspace/runtime', options.outf)
    )

    stopping_args = Stop_args(
        patience=100, max_epochs=options.train.max_epoch+1)
    early_stopping = EarlyStopping(net, **stopping_args)

    epoch = 0
    for epoch in range(options.train.max_epoch + 1):
        train()
        validate()
        test()

        writer.summarize(epoch, 'train')
        writer.summarize(epoch, 'val')
        writer.summarize(epoch, 'test')

        if early_stopping.check(
            [writer.data[epoch]['val_average_accuracy'],
             writer.data[epoch]['val_average_loss']], epoch
        ):
            writer.write('early stop\n')
            break

    writer.draw_curve(epoch)

    writer.close()
