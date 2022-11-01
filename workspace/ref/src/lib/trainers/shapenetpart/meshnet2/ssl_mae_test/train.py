''' trainer for classification with both val and test

The frame flow of the trainer.
train()
    _prepare_train()
        for epoch
            for 
                _criterion()
            _adjust_learning_rate()
            _report('val')
                _val_save()
                _validate('val')
                    _create_dict_metric()
                    for _update_dict_metric()
                    _post_process_dict_metric
                _val_update('val')
                _val_info('val')
            _report('test')
                _val_save()
                _validate('test')
                    _create_dict_metric()
                    for _update_dict_metric()
                    _post_process_dict_metric
                _val_update('test')
                _val_info('test')

author:
    zhangsihao yang

logs:
    20220919
        file created
'''
import time
from glob import glob
import os
import torch
import torch.nn.functional as F
from lib.trainers.trainer import ModelTrainer


class Trainer(ModelTrainer):
    def _prepare_train(self):
        self.to_restore = {
            'curr_epoch': -1, 'total_step': 0,
            'best_acc': -1, 'best_acc_epoch': -1
        }
        # load the model given the mode
        self._mode_load()
        # set the start and end epoch
        self.start_ep = self.to_restore['curr_epoch'] + 1
        self.end_ep = self.train_opt.max_epoch + 1

    def _criterion(self, pred, y, **kwargs):
        loss = F.cross_entropy(pred, y, reduction="mean")

        # information
        if self.to_restore['total_step'] % self.train_opt.train_freq == 0:
            batch_size = self.opt.data.train.dataloader.batch_size
            process_time = (time.time() - self.iter_start_time) / batch_size
            data_time = self.iter_start_time - self.iter_data_time

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(y.data).cpu().sum()
            acc = correct / pred.shape[0] * 100.

            info_string = (
                f'=> epoch: {self.curr_ep:5}, '
                f'time: {process_time:.3f}, data: {data_time:.3f}, '
                f'training loss is {loss.item():.3f}, '
                f'ACC: [{acc:10.5} %]'
            )

            print(info_string)
            self.logger.login(info_string)

        self.to_restore['total_step'] += 1

        return loss

    def _val_save(self):
        self.to_restore['curr_ep'] = self.curr_ep

        save_dict = {
            'to_restore': self.to_restore,
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            'train_data_provider': self.train_data_provider.get_state(),
            # 'val_data_provider': self.val_data_provider.get_state(),
            'torch_random': torch.get_rng_state(),
            'torch_cuda_state': torch.cuda.get_rng_state()
        }
        self.model_manager.save_iter(
            self.curr_ep, save_dict
        )

    def _create_dict_metric(self):
        self.dict_metric = {
            'total_correct': 0, 'total_testset': 0
        }

    def _update_dict_metric(self, pred, y, **kwargs):
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
        if split == 'val':
            if self.to_restore['best_acc'] < self.dict_metric['total_acc']:
                self.to_restore['best_acc'] = self.dict_metric['total_acc']
                self.to_restore['best_acc_epoch'] = self.curr_ep

    def _val_info(self, split):
        now = time.strftime("%c")
        now_string = f'================ {split} Acc ({now}) ================'

        print(now_string)
        self.logger.login(now_string)

        if split == 'val':
            acc = self.dict_metric['total_acc'] * 100
            ba = self.to_restore['best_acc'] * 100
            be = self.to_restore['best_acc_epoch']
            message = (
                f'epoch: {self.curr_ep:5}, '
                f'VAL  ACC: [{acc:10.5} %], '
                f'BEST ACC: [{ba:10.5} %], '
                f'BEST EP: {be:5}'
            )
        elif split == 'test':
            acc = self.dict_metric['total_acc'] * 100
            message = (
                f'epoch: {self.curr_ep:5}, '
                f'TEST ACC: [{acc:10.5} %], '
            )
            # self.search_clear_ckpt()
        else:
            raise ValueError('Invalid split')

        print(message)
        self.logger.login(message)

    def _adjust_learning_rate(self):
        pass

    def search_clear_ckpt(self):
        ckpt_path_pattern = os.path.join(
            self.model_manager.root, '*.pth'
        )
        list_ckpt_path = glob(ckpt_path_pattern)
        best_ckpt_path = os.path.join(
            self.model_manager.root,
            f"{self.model_manager.prefix}_{self.to_restore['best_acc_epoch']}.pth"
        )
        for ckpt_path in list_ckpt_path:
            if ckpt_path != best_ckpt_path:
                os.system(f'rm {ckpt_path}')
