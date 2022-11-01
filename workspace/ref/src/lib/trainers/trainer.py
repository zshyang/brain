''' trainer

class:
    c00 ModelTrainer
        00  __init__(self, logger, tvtdp, net, optimizer, scheduler, model_manager)
        01  _load_net(self)
        02  _search_latest_ckpt(self)
        03  _load_pretrain(self)
        04  _mode_load(self)
        05  _prepare_train(self)
        06  _criterion(self, pred, target)
        07  _dict_cuda(self, dict_tensor)
        08  _cuda(self, in_data, target)
            <-- 07  _dict_cuda
        09  _create_dict_metric(self)
        10  _update_dict_metric(self, pred, **target)
        11  _post_process_dict_metric(self)
        12  _validate(self, split)
            <-- 09  _create_dict_metric
            <-- 08  _cuda
            <-- 10  _update_dict_metric
            <-- 11  _post_process_dict_metric
        13  _val_update(self, split)
        14  _val_info(self, split)
        15  _val_save(self)
        16  _nonempty_split(self, split:str)->bool
        17  _report(self, split: str)
            <-- 16  _nonempty_split
            <-- 15  _val_save
            <-- 12  _validate
            <-- 13  _val_update
            <-- 14  _val_info
        18  _adjust_learning_rate(self)
        19  train(self, opt)
            <-- 05  _prepare_train
            <-- 08  _cuda
            <-- 06  _criterion
            <-- 18  _adjust_learning_rate
            <-- 17  _report
            <-- 20  _report_time
        20  _report_time(self)

name convention:
    le = load epoch
    st = start time
    tl = time limit
    ts = time scale
    sat = save at test
    sav = save at validation
    trt = train time
    tst = test time
    vlt = validation time
    etne = estimated time for next epoch
    ttst = total start time
    tvtdp = train validation test data provider

author:
    zhangsihao yang

date:
    20220310

logs:
    20220628
        add minor modification
    20220823
        add documentation
    20220911
        add debug option
'''
import os
import sys
import time
from glob import glob

import torch
import torch.nn.functional as F


class ModelTrainer():
    def __init__(self, logger, tvtdp, net, optimizer, scheduler, model_manager):

        self.logger = logger

        self.tvtdp = tvtdp
        self.train_data_provider = tvtdp.trdp
        self.val_data_provider = tvtdp.vldp
        self.test_data_provider = tvtdp.tsdp

        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_manager = model_manager

        self.curr_ep = 0

        if self.val_data_provider is not None:
            self.sav = True
            self.sat = False
        elif self.test_data_provider is not None:
            self.sav = False
            self.sat = True
        else:
            # raise Exception('val and test are both None')
            print('val and test are both None')
            self.sav = False
            self.sat = False

        self.trt = 0
        self.vlt = 0
        self.tst = 0

    def _load_net(self):
        ''' under mode 2 to restore 
        the pretrained weights
        '''
        self.model_manager.load_ep(
            self.train_opt.params_2.load_epoch,
            run_variables=self.to_restore,
            net=self.net,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            tvtdp=self.tvtdp,
        )

    def _search_latest_ckpt(self):
        root = os.path.join(
            '/runtime', self.opt.outf
        )
        pattern = os.path.join(
            root, '*.pth'
        )
        list_ckpt = glob(pattern)
        if len(list_ckpt) <= 0:
            print('=> empty folder')
            return -1
        list_ckpt.sort()
        list_ckpt.sort(key=len)
        # get the number from the path
        name = list_ckpt[-1]
        name = name.split('/')[-1]
        name = name[
            len(
                self.opt.manager.ckpt_prefix
            )+1:-4
        ]
        return int(name)

    def _load_pretrain(self):
        self.model_manager.load_pretrain(
            self.net,
            self.train_opt.params_1.ckpt_name
        )

    def _mode_load(self):
        if self.train_opt.mode == 0:
            print('=> train from scratch')
        elif self.train_opt.mode == 1:
            print('=> train from pretrain')
            self._load_pretrain()
        elif self.train_opt.mode == 2:
            print('=> train from restore')
            self._load_net()
        elif self.train_opt.mode == 3:
            le = self._search_latest_ckpt()
            if le == -1:
                print('=> train from scratch')
            else:
                self.train_opt.params_2.load_epoch = le
                self._load_net()
        else:
            raise Exception('unknown loading mode')

    def _prepare_train(self):
        ''' prepare meta information for training
        1. create a dict for restoring training next time
        2. load the model given the training mode
        3. set start epoch and end epoch
        '''
        raise NotImplementedError(
            '_prepare_train!'
        )

    def _criterion(self, pred, target):
        ''' compute loss given the pred and target
        '''
        raise NotImplementedError(
            '_criterion!'
        )

    def _dict_cuda(self, dict_tensor):
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

    def _cuda(self, in_data, target):
        ''' move input data and 
        target onto gpu
        '''
        self._dict_cuda(in_data)
        self._dict_cuda(target)

    def _create_dict_metric(self):
        raise NotImplementedError(
            '_create_dict_metric!'
        )
    
    def _update_dict_metric(self, pred, **target):
        raise NotImplementedError(
            '_update_dict_metric!'
        )

    def _post_process_dict_metric(self):
        ''' TODO: what is the purpose of this function
        '''
        raise NotImplementedError(
            '_post_process_dict_metric!'
        )

    def _validate(self, split):
        self.net.eval()

        self._create_dict_metric()

        if split == 'val':
            dp = self.val_data_provider
        elif split == 'test':
            dp = self.test_data_provider
        else:
            raise Exception('unknown split')

        with torch.no_grad():
            for i in range(len(dp)):
                in_data, target = dp.get_next_batch()
                self._cuda(in_data, target)
                pred = self.net(**in_data)
                torch.cuda.empty_cache()

                self._update_dict_metric(pred, **target)

                if getattr(self.opt.train, 'debug', False):
                    break

            self._post_process_dict_metric()

        self.net.train()

    def _val_update(self, split):
        raise NotImplementedError('_val_update!')

    def _val_info(self, split):
        raise NotImplementedError('_val_info!')

    def _val_save(self):
        raise NotImplementedError('_val_save!')

    def _nonempty_split(self, split:str)->bool:
        if split == 'val':
            return self.val_data_provider is not None
        elif split == 'test':
            return self.test_data_provider is not None
        else:
            raise Exception('unknown split')

    def _report(self, split: str):
        ''' generate a report

        Args:
            split: either be val or test
        '''
        if self._nonempty_split(split):
            if split == 'val' and self.sav:
                self._val_save()
            if split == 'test' and self.sat:
                self._val_save()
            self._validate(split)
            self._val_update(split)
            self._val_info(split)

    def _adjust_learning_rate(self):
        raise NotImplementedError(
            '_adjust_learning_rate!'
        )

    def _pre_train(self):
        pass

    def _post_train(self):
        pass

    def train(self, opt):
        ''' 
        name convention:
            st = start time
            trt = train time
            tst = test time
            vlt = validation time
            ttst = total start time
        '''
        self.ttst = time.time()
        self.logger.login('=> begin training')

        self.train_opt = opt.train
        self.opt = opt

        loss_params = opt.train.loss_params

        self._prepare_train()

        for i in range(self.start_ep, self.end_ep):
            self.curr_ep = i

            self.st = time.time()
            self.iter_data_time = time.time()

            self._pre_train()
            for _ in range(len(self.train_data_provider)):
                self.iter_start_time = time.time()

                self.net.train()
                in_data, target = self.train_data_provider.get_next_batch()
                self._cuda(in_data, target)
                pred = self.net(**in_data)
                loss = self._criterion(pred, **target, **loss_params)
                if torch.isnan(loss):
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                torch.cuda.empty_cache()

                self.iter_data_time = time.time()

                if getattr(self.opt.train, 'debug', False):
                    break
            self._post_train()

            self.trt = int(time.time() - self.st)

            # scheduler step
            if i % self.train_opt.scheduler_step == 0:
                if self.scheduler is not None:
                    self.scheduler.step()
                else:
                    self._adjust_learning_rate()

            # save and validate
            if i % self.train_opt.val_freq == 0:
                st = time.time()
                self._report('val')
                self.vlt = int(time.time() - st)

            # test
            if i % self.train_opt.test_freq == 0:
                st = time.time()
                self._report('test')
                self.tst = int(time.time() - st)

            self._report_time()

        # remove the flag file
        print('=> finish the training!')
        exit_file = os.path.join(
            '/runtime', self.opt.outf, 'cont.txt'
        )
        if os.path.exists(exit_file):
            os.system(f'rm {exit_file}')

    def _report_time(self):
        p_string = f'=> epoch {self.curr_ep} ' + \
                   f'train time: {self.trt} ' + \
                   f'val time: {self.vlt} ' + \
                   f'test time: {self.tst}'
        print(p_string)
        self.logger.login(p_string)

        # timeout the program
        ts = self.opt.get('timescale', -1)
        etne = (self.trt + self.vlt + self.tst) * ts
        tl = self.opt.get('timelimit', -1)
        if tl == -1:
            return
        elif ((time.time() - self.ttst) + etne) >= tl:
            print('=> time up and exit!')
            exit_file = os.path.join(
                '/runtime', self.opt.outf, 'cont.txt'
            )
            os.system(f'touch {exit_file}')
            sys.exit(0)


class ModelNetTrainer(ModelTrainer):
    def _prepare_train(self):
        ''' prepare for training
        1. set the to_restore variabel
        2. load pre-trained model if asked
        3. set the start epoch and end epoch
        '''
        self.to_restore = {
            'curr_ep': -1, 'best_metric': 0,
            'best_ep': 0}

        # load pre-trained model if asked
        if self.train_opt.mode == 1:
            self._load_pretrain()
        elif self.train_opt.mode == 2:
            self._load_net(self.to_restore)
        elif self.train_opt.mode == 0:
            pass
        else:
            raise Exception('unknown train mode')

        self.start_ep = self.to_restore['curr_ep'] + 1
        self.end_ep = self.train_opt.max_epoch + 1

    def _criterion(self, pred, labels, label_smoothing):
        """Calculate cross entropy loss, apply label smoothing 
        if needed."""

        pred = pred[0]
        labels = labels.contiguous().view(-1)
        if label_smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(
                1, labels.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + \
                (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(
                pred, labels, reduction="mean")
        return loss

    def _create_dict_metric(self):
        ''' create a dictionary of metrics to be reported
        '''
        self.dict_metric = {
            'total_correct': 0, 'total_testset': 0
        }

    def _update_dict_metric(self, pred, labels):
        ''' update the dictionary of metrics according to 
        the predictions.
        '''
        pred = pred[0]
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(labels.data).cpu().sum()
        self.dict_metric['total_correct'] += correct.item()
        self.dict_metric['total_testset'] += pred.size()[0]

    def _post_process_dict_metric(self):
        self.dict_metric['total_acc'] = \
            self.dict_metric['total_correct'] / \
            float(self.dict_metric['total_testset'])

    def _val_update(self, split):
        ''' update to_restore during validation
        '''
        best = self.to_restore['best_metric']
        curr = self.dict_metric['total_acc']
        if  best < curr:
            self.to_restore['best_metric'] = curr
            self.to_restore['best_ep'] = self.curr_ep

    def _val_info(self, split):
        ep_string = f'Epoch {self.curr_ep:8d}: '

        best = self.to_restore['best_metric']
        curr = self.dict_metric['total_acc']
        best_ep = self.to_restore['best_ep']

        testa_string = f'{split} accuracy {curr:.4f} '
        besta_string = f'Best accuracy {best:4f} at epoch {best_ep:8d}'

        p_string = ep_string + testa_string + besta_string
        print(p_string)
        self.logger.login(p_string)

    def _val_save(self):
        save_dict = {
            "curr_ep": self.curr_ep,
            'best_metric': self.to_restore['best_metric'],
            'best_ep': self.to_restore['best_ep'],
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'tvtdp': self.tvtdp.get_state(),
        }
        self.model_manager.save_iter(
            self.curr_ep, save_dict
        )


class PointNetVAETrainer(ModelTrainer):
    def _prepare_train(self, opt):
        # best_metric here is the best loss value we have
        self.to_restore = {
            'curr_ep': -1, 'best_metric': 9999,
            'best_ep': 0,
        }

        # load pre-trained model if asked
        if self.train_opt.load:
            self._load_net(self.to_restore)

        self.start_ep = self.to_restore['curr_ep'] + 1
        self.end_ep = self.train_opt.max_epoch + 1

    def _criterion(self, pred, y, is_vae, kl_weight):
        ''' Compute the reconstruction loss and KLD if 
        needed.
        Args:
            y is the output point cloud (B, N, 3)
        '''

        y_, mean, log_var = pred

        # compute chamfer distance
        import pytorch3d.loss
        loss, _ = pytorch3d.loss.chamfer_distance(
            x=y_, y=y)

        # compute KLD is needed
        if is_vae:
            KLD = - 0.5 * torch.mean(
                1 + log_var - mean.pow(2) - log_var.exp())
            loss = loss + kl_weight * KLD            

        return loss

    def _create_dict_metric(self):
        ''' create a dictionary of metrics to be reported
        '''
        self.dict_metric = {
            'chamfer_dist': [], 'kld': []}

    def _update_dict_metric(self, pred, y):
        ''' update the dictionary of metrics according to 
        the predictions.
        '''
        import pytorch3d.loss
        ch_loss, _ = pytorch3d.loss.chamfer_distance(
            x=pred[0], y=y, batch_reduction=None)
        kld = []
        if (pred[1] is not None) and (pred[2] is not None):
            mean = pred[1]
            log_var = pred[2]
            kld = - 0.5 * torch.mean(
                1 + log_var - mean.pow(2) - log_var.exp(), 1)
            kld = kld.data.cpu().numpy().tolist()            
        ch_loss = ch_loss.data.cpu().numpy().tolist()
        self.dict_metric['chamfer_dist'].extend(ch_loss)
        self.dict_metric['kld'].extend(kld)
    
    def _post_process_dict_metric(self):
        list_chamfer = self.dict_metric['chamfer_dist']
        list_kld = self.dict_metric['kld']
        self.dict_metric['mean_chamfer'] = \
            sum(list_chamfer) / len(list_chamfer)
        if self.train_opt.loss_params.is_vae:
            self.dict_metric['mean_kld'] = \
                sum(list_kld) / len(list_kld)

    def _val_update(self):
        ''' update to_restore during validation
        '''
        best = self.to_restore['best_metric']
        curr = self.dict_metric['mean_chamfer']
        if  best > curr:
            self.to_restore['best_metric'] = curr
            self.to_restore['best_ep'] = self.curr_ep

    def _val_info(self):
        ep_string = f'Epoch {self.curr_ep:8d}: '
        best = self.to_restore['best_metric']
        curr = self.dict_metric['mean_chamfer']
        best_ep = self.to_restore['best_ep']
        testa_string = f'Test chamfer {curr:.4f} '
        besta_string = f'Best chamfer {best:4f} at epoch {best_ep:8d}'
        p_string = ep_string + testa_string + besta_string
        print(p_string)
        self.logger.login(p_string)

    def _val_save(self):
        save_dict = {
            "curr_ep": self.curr_ep,
            'best_metric': self.to_restore['best_metric'],
            'best_ep': self.to_restore['best_ep'],
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'train_data_provider': self.train_data_provider.get_state(),
            'test_data_provider': self.test_data_provider.get_state(),
            'torch_random': torch.get_rng_state(),
            'torch_cuda_state': torch.cuda.get_rng_state()}
        self.model_manager.save_iter(
            self.curr_ep, save_dict)


'''
Zhangsihao Yang
04/04/2022

lc = loss for coarse point cloud
lf = loss for fine point cloud
ts = total steps
le = load epoch
'''
def piecewise_constant(global_step, boundaries, values):
    """substitute for tf.train.piecewise_constant:
    https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/piecewise_constant
    global_step can be either training epoch or training step
    """
    if len(boundaries) != len(values) - 1:
        raise ValueError(
            "The length of boundaries should be 1 less than the length of values")

    if global_step <= boundaries[0]:
        return values[0]
    elif global_step > boundaries[-1]:
        return values[-1]
    else:
        for low, high, v in zip(
            boundaries[:-1], boundaries[1:], values[1:-1]
        ):
            if (global_step > low) & (global_step <= high):
                return v


class OcCoTrainer(ModelTrainer):
    def _prepare_train(self):
        # best_metric here is the best loss value we have
        self.to_restore = {
            'curr_ep': -1, 'best_lf': 9999, 'best_lc': 9999,
            'best_lfep': 0, 'best_lcep': 0, 'ts': 0,
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

    def _criterion(self, pred, y, **kwargs):
        ''' Compute the reconstruction loss and KLD if 
        needed.
        Args:
            y is the output point cloud (B, N, 3)
        '''

        coarse, fine = pred
        import pytorch3d.loss
        lc, _ = pytorch3d.loss.chamfer_distance(
            x=coarse, y=y
        )
        lf, _ = pytorch3d.loss.chamfer_distance(
            x=fine, y=y
        )
        step = self.curr_ep * len(self.train_data_provider)
        alpha = piecewise_constant(
            step, [10000, 20000, 50000],
            [0.01, 0.1, 0.5, 1.0]
        )
        loss = lc + alpha * lf

        if self.to_restore['ts'] % self.train_opt.train_freq == 0:
            print(f'=> training loss is {loss} alpha is {alpha}')
        self.to_restore['ts'] += 1

        return loss

    def _create_dict_metric(self):
        self.dict_metric = {
            'total_number': 0, 'total_lf': 0,
            'total_lc': 0,
        }

    def _update_dict_metric(self, pred, y):
        ''' update the dictionary of metrics according to 
        the predictions.
        '''
        coarse, fine = pred
        import pytorch3d.loss
        lc, _ = pytorch3d.loss.chamfer_distance(
            x=coarse, y=y
        )
        lf, _ = pytorch3d.loss.chamfer_distance(
            x=fine, y=y
        )
        lc = lc.data.cpu().item()
        lf = lf.data.cpu().item()
        self.dict_metric['total_number'] += 1
        self.dict_metric['total_lc'] += lc
        self.dict_metric['total_lf'] += lf

    def _post_process_dict_metric(self, pred, y, x):
        self.dict_metric['avg_lf'] = \
            self.dict_metric['total_lf'] / \
            float(self.dict_metric['total_number'])
        self.dict_metric['avg_lc'] = \
            self.dict_metric['total_lc'] / \
            float(self.dict_metric['total_number'])

        # visualization # TODO need to be revise the logic
        coarse, fine = pred
        all_pcds = [
            item.detach().cpu().numpy() for item in [
                x.transpose(2, 1), coarse, fine, y
            ]
        ]
        from lib.vis.pc_viewer import plot_pcd_three_views
        for i in range(coarse.shape[0]):
            plot_path = os.path.join(
                '/runtime', self.opt.outf, 'plots',
                f'epoch_{self.curr_ep}_step_' + \
                f"{self.to_restore['ts']}_{i}.png"
            )
            os.makedirs(
                os.path.dirname(plot_path), exist_ok=True
            )
            pcds = [x[i] for x in all_pcds]
            plot_pcd_three_views(
                plot_path, pcds,
                [
                    'input', 'coarse output', 'fine output',
                    'ground truth'
                ]
            )

    def _val_update(self, split):
        ''' update to_restore during validation
        '''
        blf = self.to_restore['best_lf']
        currlf = self.dict_metric['avg_lf']
        if  blf > currlf:
            self.to_restore['best_lf'] = currlf
            self.to_restore['best_lfep'] = self.curr_ep

        blc = self.to_restore['best_lc']
        currlc = self.dict_metric['avg_lc']
        if  blc > currlc:
            self.to_restore['best_lc'] = currlc
            self.to_restore['best_lcep'] = self.curr_ep

    def _val_info(self, split):
        ep_string = f'Epoch {self.curr_ep:8d}: '

        blc = self.to_restore['best_lc']
        blf = self.to_restore['best_lf']
        currlc = self.dict_metric['avg_lc']
        currlf = self.dict_metric['avg_lf']
        best_lfep = self.to_restore['best_lfep']
        best_lcep = self.to_restore['best_lcep']

        testa_string = f'{split} fine loss: {currlf:.6f} ' + \
                       f'coarse loss: {currlc:.6f}\n'
        besta_string = f'Best lf {blf:.4f} at epoch {best_lfep:8d}\n' + \
                       f'Best lc {blc:.4f} at epoch {best_lcep:8d}'

        p_string = ep_string + testa_string + besta_string
        print(p_string)
        self.logger.login(p_string)

    def _val_save(self):
        self.to_restore['curr_ep'] = self.curr_ep
        save_dict = {
            'to_restore': self.to_restore,
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'tvtdp': self.tvtdp.get_state(),
        }
        self.model_manager.save_iter(
            self.curr_ep, save_dict
        )
