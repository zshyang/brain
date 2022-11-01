''' trainer for classification

The frame flow of the trainer.
ModelTrainer._report()
    _val_save()
    _validate()
        _create_dict_metric()
        for _update_dict_metric()
        _post_process_dict_metric
    _val_update()
    _val_info

class:
    c00 Trainer(ModelTrainer)
        00  _prepare_train(self)
        01 _criterion(self, pred, y, **kwargs)
        02 _val_save(self)
        03 _create_dict_metric(self)
        04 _update_dict_metric(self, pred, y)
        05 _post_process_dict_metric(self)
        06 _val_update(self, split)
        07 _val_info(self, split)
        08 _adjust_learning_rate(self)

author:
    Zhangsihao Yang

date:
    20220628
        c00-00,01
    20220822
        c00-02,03,04,05,06,07,08
'''
import time

import torch.nn.functional as F
from lib.trainers.trainer import ModelTrainer


class Trainer(ModelTrainer):
    def _prepare_train(self):
        '''
        name convention:
            ba = best accuracy
            ts = total steps
            bae = best accuracy epoch
        '''
        # TODO: add per-class accuracy, class weighted accuracy, and confusion matrix
        self.to_restore = {
            'curr_ep': -1, 'ts': 0, 'ba': -1, 'bae': -1
        }
        # load the model given the mode
        self._mode_load()
        # set the start and end epoch
        self.start_ep = self.to_restore['curr_ep'] + 1
        self.end_ep = self.train_opt.max_epoch + 1

    def _criterion(self, pred, y, **kwargs):
        label_smoothing = kwargs.get('label_smoothing', None)
        # pred = pred[0]
        # labels = labels.contiguous().view(-1)
        if label_smoothing:
            eps = 0.2
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(
                1, y.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + \
                (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, y, reduction="mean")

        if self.to_restore['ts'] % self.train_opt.train_freq == 0:
            batch_size = self.opt.data.train.dataloader.batch_size
            process_time = (time.time() - self.iter_start_time) / batch_size
            data_time = self.iter_start_time - self.iter_data_time
            info_string = (
                f'=> epoch: {self.curr_ep:5}, '
                f'time: {process_time:.3f}, data: {data_time:.3f}, '
                f'training loss is {loss.item():.3f}'
            )
            print(info_string)
            self.logger.login(info_string)
        self.to_restore['ts'] += 1

        return loss

    def _val_save(self):
        pass

    def _create_dict_metric(self):
        self.dict_metric = {
            'total_correct': 0, 'total_testset': 0
        }

    def _update_dict_metric(self, pred, y):
        ''' update the dictionary of metrics according to 
        the predictions.
        '''
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(y.data).cpu().sum()
        self.dict_metric['total_correct'] += correct.item()
        self.dict_metric['total_testset'] += pred.size()[0]

    def _post_process_dict_metric(self):
        self.dict_metric['total_acc'] = \
            self.dict_metric['total_correct'] / \
            float(self.dict_metric['total_testset'])

    def _val_update(self, split):
        '''
        ba = best accuracy
        '''
        if self.to_restore['ba'] < self.dict_metric['total_acc']:
            self.to_restore['ba'] = self.dict_metric['total_acc']

    def _val_info(self, split):
        '''
        ba = best accuracy
        '''
        now = time.strftime("%c")
        now_string = f'================ Testing Acc ({now}) ================'

        print(now_string)
        self.logger.login(now_string)

        acc = self.dict_metric['total_acc'] * 100
        ba = self.to_restore['ba'] * 100
        message = (
            f'epoch: {self.curr_ep:5}, '
            f'TEST ACC: [{acc:10.5} %], '
            f'BEST ACC: [{ba:10.5} %]'
        )

        print(message)
        self.logger.login(message)

    def _adjust_learning_rate(self):
        pass
