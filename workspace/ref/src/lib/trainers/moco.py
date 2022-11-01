''' trainer for moco

author
    Zhangsihao Yang

date
    2022-0408

name convention
    al = average loss
    at1a = average top 1 accuracy
    at5a = average top 5 accuracy
    bl = best loss
    ble = best loss epoch
    bt1a = best top 1 accuracy
    bt1ae = best top 1 accuracy epoch
    bt5a = best top 5 accuracy
    bt5ae = best top 5 accuracy epoch
'''
import math

import torch
import torch.nn as nn
from lib.trainer.trainer import ModelTrainer


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

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(
                -1
            ).float().sum(0, keepdim=True)
            res.append(
                correct_k.mul_(
                    100.0 / batch_size
                )
            )
        return res


class MoCoTrainer(ModelTrainer):
    def _prepare_train(self):
        # best_metric here is the best loss value we have
        self.to_restore = {
            'curr_ep': -1, 'ts': 0,
            'bt1a': 0, 'bt1ae': 0,
            'bt5a': 0, 'bt5ae': 0,
            'bl': 99999, 'ble': 0,
        }

        # load pre-trained model if asked
        if self.train_opt.mode == 0:
            print('=> train from scritch')
        elif self.train_opt.mode == 2:
            self._load_net()
        elif self.train_opt.mode == 3:
            le = self._search_latest_ckpt()
            if le == -1:
                print('=> train from scritch')
            else:
                self.train_opt.params_2.load_epoch = le
                self._load_net()

        self.start_ep = self.to_restore['curr_ep'] + 1
        self.end_ep = self.train_opt.max_epoch + 1

        # loss function
        self.criterion = nn.CrossEntropyLoss().cuda()

    def _criterion(self, pred, **kwargs):
        loss = self.criterion(pred[0], pred[1])

        if self.to_restore['ts'] % \
            self.train_opt.train_freq == 0:
            print(f'=> training loss is {loss}')
        self.to_restore['ts'] += 1

        return loss

    def _create_dict_metric(self):
        self.dict_metric = {
            'total_number': 0,
            'total_loss': 0,
            'total_top1_correct': 0,
            'total_top5_correct': 0,
        }

    def _update_dict_metric(self, pred, **kwargs):
        output, target = pred
        loss = self.criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier
        # accuracy measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        self.dict_metric['total_number'] += 1
        self.dict_metric['total_loss'] += loss.item()
        self.dict_metric['total_top1_correct'] += \
            acc1.item()
        self.dict_metric['total_top5_correct'] += \
            acc5.item()

    def _post_process_dict_metric(self, pred, **kwargs):
        self.dict_metric['avg_loss'] = \
            self.dict_metric['total_loss'] / \
            float(self.dict_metric['total_number'])
        self.dict_metric['avg_top1_acc'] = \
            self.dict_metric['total_top1_correct'] / \
            float(self.dict_metric['total_number'])
        self.dict_metric['avg_top5_acc'] = \
            self.dict_metric['total_top5_correct'] / \
            float(self.dict_metric['total_number'])

    def _val_update(self, split):
        '''
        update self.to_restore given self.dict_metric
        '''
        bt1a = self.to_restore['bt1a']
        bt1ae = self.to_restore['bt1ae']
        bt5a = self.to_restore['bt5a']
        bt5ae = self.to_restore['bt5ae']
        bl = self.to_restore['bl']
        ble = self.to_restore['ble']

        al = self.dict_metric['avg_loss']
        at1a = self.dict_metric['avg_top1_acc']
        at5a = self.dict_metric['avg_top5_acc']

        if bl > al:
            self.to_restore['bl'] = al
            self.to_restore['ble'] = self.curr_ep
        if bt1a < at1a:
            self.to_restore['bt1a'] = at1a
            self.to_restore['bt1ae'] = self.curr_ep
        if bt5a < at5a:
            self.to_restore['bt5a'] = at5a
            self.to_restore['bt5ae'] = self.curr_ep

    def _val_info(self, split):
        ep_string = f'Epoch {self.curr_ep:8d}:\n'

        bt1a = self.to_restore['bt1a']
        bt1ae = self.to_restore['bt1ae']
        bt5a = self.to_restore['bt5a']
        bt5ae = self.to_restore['bt5ae']
        bl = self.to_restore['bl']
        ble = self.to_restore['ble']

        al = self.dict_metric['avg_loss']
        at1a = self.dict_metric['avg_top1_acc']
        at5a = self.dict_metric['avg_top5_acc']

        string_info = (
            f'current loss {al:.6f} \t'
            f'current top 1 accuracy {at1a:.3f} \t'
            f'current top 5 accuracy {at5a:.3f}\n'
            f'best loss {bl:.6f} at {ble:8d}\n'
            f'best top 1 accuracy {bt1a:.3f} '
            f'at {bt1ae}\n'
            f'best top 5 accuracy {bt5a:.3f} '
            f'at {bt5ae} '
        )

        p_string = ep_string + string_info
        print(p_string)
        self.logger.login(p_string)

    def _val_save(self):
        self.to_restore['curr_ep'] = self.curr_ep
        save_dict = {
            'to_restore': self.to_restore,
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            # "scheduler": self.scheduler.state_dict(),
            'tvtdp': self.tvtdp.get_state(),
        }
        self.model_manager.save_iter(
            self.curr_ep, save_dict
        )

    def _adjust_learning_rate(self):
        """Decay the learning rate based on schedule"""
        lr = self.opt.optim.params.lr
        # cosine lr schedule
        lr *= 0.5 * (
            1. + math.cos(
                math.pi * self.curr_ep / self.end_ep
            )
        )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class DEMCTrainer(MoCoTrainer):
    ''' DisEntangle MoCo trainer
    '''
    def _criterion(
        self, pred, day, **kwargs
    ):
        loss = self.criterion(pred[0], pred[1])

        if self.to_restore['ts'] % \
            self.train_opt.train_freq == 0:
            print(f'=> training loss is {loss}')
        self.to_restore['ts'] += 1

        return loss

