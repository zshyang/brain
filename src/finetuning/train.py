''' fine tuning on the pre-trained model.

author: 
    zhangsihao yang

logs:
    20221106 file created
'''
import argparse
import os
import random
import time

import numpy as np
import pytorch3d.loss
import torch
import torch.nn as nn
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
    ckpt = torch.load(ckpt_path)
    nd = net.state_dict()
    pd = ckpt['net']
    pd = {
        k: v for k, v in pd.items() if \
        k in nd and pd[k].size() == nd[k].size()
    }

    print('=' * 27)
    print('Restored Params and Shapes:')
    for k, v in pd.items():
        print(k, ': ', v.size())
    print('=' * 68)

    nd.update(pd)
    net.load_state_dict(nd)

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
        num_workers=int(opt.dataloader.num_workers),
        shuffle=bool(opt.dataloader.shuffle),
    )
    return dataloader


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


def accuracy(output, target, topk=(1,)):
    """ Computes the accuracy over the k top predictions
    for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(
            target.view(1, -1).expand_as(pred)
        )

        res = correct.data.detach().cpu().numpy().astype(int).tolist()[0]

        return res


def _compute_acc(net, dataloader):
    list_acc = []
    for in_data, target in dataloader:
        _cuda(in_data, target)
        pred, fea = net(**in_data)
        acc = accuracy(pred, target['label'])
        list_acc.extend(acc)

        if options.debug:
            break

    return np.mean(list_acc)


def finetune():
    val_info = {'val_acc': 0.0, 'val_ep': -1}
    test_info = {'test_acc': 0.0}

    for i in range(options.train.max_epoch + 1):
        net.train()

        for in_data, target in train_dataloader:

            _cuda(in_data, target)
            pred, fea = net(**in_data)
            loss = criterion(pred, target['label'])

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            if options.debug:
                break

        if scheduler is not None:
            scheduler.step()

        # validate
        net.eval()
        curr_val_acc = _compute_acc(net, val_dataloader)

        if curr_val_acc >= val_info['val_acc']:

            val_info['val_acc'] = curr_val_acc
            val_info['val_ep'] = i

            test_acc = _compute_acc(net, test_dataloader)

            if curr_val_acc == val_info['val_acc']:
                test_info['test_acc'] = max(
                    test_info['test_acc'], test_acc
                )
            else:
                test_info['test_acc'] = test_acc

            print('Update: ', val_info, test_info)
            str_log = f'update: {val_info} {test_info}\n'
            log.write(str_log)
            log.flush()

    print('=> finish the training!')
    print(test_info)
    log.write(str(test_info))
    log.flush()


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

    #======== net, loss, optimizer, scheduler ========
    net = Network(**options.model.params)
    # load weight from pre-trained model
    net = load_net_from_pre_trained(net, '/runtime/300.pt')
    net = net.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = getattr(optim, options.optim.name)(
        net.parameters(), **options.optim.params
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

    #======== log =======
    log_path = os.path.join(
        '/runtime', options.outf, 
        f'{options.logger.name}_log.txt'
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if options.logger.overwrite:
        log = open(log_path, 'w')
    else:
        log = open(log_path, 'a')

    #======== finetune ========
    finetune()

    log.close()
