''' trainer for segmentation

author
    Zhangsihao Yang

date
    04/23/2022

name convention:
    a = accuracy
    aa = avarage accuracy
    al = avarage loss
    ba = best accuracy
    bae = best accuracy epoch
    tm = tranlation matrix
'''
import torch.nn.functional as F
from lib.model.pointnet.util import feature_transform_regularizer
from lib.trainer.trainer import ModelTrainer
from lib.trainer.util import accuracy


class SegTrainer(ModelTrainer):
    def _prepare_train(self):
        self.to_restore = {
            'curr_ep': -1, 
            'ts': 0,
            'ba': 0,
            'bae': 0
        }

        self._mode_load()

        self.start_ep = \
          self.to_restore['curr_ep'] + \
          1
        self.end_ep = \
          self.train_opt.max_epoch + \
          1

    def _criterion(
        self, pred, 
        mat_diff_loss_scale, y,
        **kwargs,
    ):
        x, tm1 = pred
        # x : [b, n_p, 27]
        # tm1 : [b, 64, 64]

        loss = F.nll_loss(x, y)
        mat_diff_loss = \
          feature_transform_regularizer(
              tm1
          )
        loss = loss + mat_diff_loss * \
          mat_diff_loss_scale

        # print the loss and training
        # accuracy
        if self.to_restore['ts'] % \
          self.train_opt.train_freq == \
          0:
            print(
                '=> training loss is '
                f'{loss:.7f}'
            )

            a = accuracy(x, y)
            print(
                '=> training acc is '
                f'{a[0].item():.2f}%'
            )

        self.to_restore['ts'] += 1

        return loss

    def _val_save(self):
        self.to_restore['curr_ep'] = \
          self.curr_ep
        save_dict = {
            'to_restore':
            self.to_restore,
            'net':
            self.net.state_dict(),
            'optimizer': 
            self.optimizer.state_dict(),
            'tvtdp':
            self.tvtdp.get_state(),
            'scheduler':
            self.scheduler.state_dict(),
        }
        self.model_manager.save_iter(
            self.curr_ep, save_dict
        )

    def _create_dict_metric(self):
        self.dict_metric = {
            'n': 0,
            'l': 0,
            'mdl': 0,
            'a': 0,
        }

    def _update_dict_metric(
        self, pred, y, **kwargs
    ):
        x, tm1 = pred
        # x : [b, n_p, 27]
        # tm1 : [b, 64, 64]

        l = F.nll_loss(x, y)
        mdl = \
          feature_transform_regularizer(
              tm1
          )

        a = accuracy(x, y)
        
        self.dict_metric['n'] += 1
        self.dict_metric['l'] += \
          l.item()
        self.dict_metric['mdl'] += \
          mdl.item()
        self.dict_metric['a'] += \
          a[0].item()
    
    def _post_process_dict_metric(
        self, pred, **kwargs
    ):
        self.dict_metric['al'] = \
          self.dict_metric['l'] / \
          float(self.dict_metric['n'])
        self.dict_metric['amdl'] = \
          self.dict_metric['mdl'] / \
          float(self.dict_metric['n'])
        self.dict_metric['aa'] = \
          self.dict_metric['a'] / \
          float(self.dict_metric['n'])
    
    def _val_update(self, split):
        ba = self.to_restore['ba']
        aa = self.dict_metric['aa']
        
        if ba < aa:
            self.to_restore['ba'] = aa
            self.to_restore['bae'] = \
              self.curr_ep

    def _val_info(self, split):
        ep_string = (
            'Epoch '
            f'{self.curr_ep:8d}:\n'
        )

        ba = self.to_restore['ba']
        bae = self.to_restore['bae']
        aa = self.dict_metric['aa']
        al = self.dict_metric['al']
        amdl = self.dict_metric['amdl']

        string_info = (
            f'current loss {al:.6f}\t'
            f'current acc {aa:.3f}\t'
            f'current mdl {amdl:.6f}\n'
            f'best acc {ba:.3f}\t'
            f'at {bae}'
        )

        p_string = ep_string + \
          string_info
        print(p_string)
        self.logger.login(p_string)
