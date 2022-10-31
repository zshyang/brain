''' train the model with single gpu.

author:
    zhangsihao yang

logs:
    20221030    file created
'''
import argparse
import random
import time

import numpy as np
import pytorch3d.loss
import torch

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

# from lib.trainers.trainer import ModelTrainer


class PointNetVAETrainer(ModelTrainer):
    def _prepare_train(self):
        # best_metric here is the best loss value we have
        self.to_restore = {
            'curr_ep': -1, 'best_metric': 9999,
            'best_ep': 0, 'total_step': 0
        }

        self._mode_load()

        self.start_ep = self.to_restore['curr_ep'] + 1
        self.end_ep = self.train_opt.max_epoch + 1

    def _criterion(self, pred, y, is_vae=False, kl_weight=0., **kwargs):
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

        # compute KLD is needed
        if is_vae:
            KLD = - 0.5 * torch.mean(
                1 + log_var - mean.pow(2) - log_var.exp())
            loss = loss + kl_weight * KLD

        # show and record the loss information
        if self.to_restore['total_step'] % self.train_opt.train_freq == 0:
            batch_size = self.opt.data.train.dataloader.batch_size
            process_time = (time.time() - self.iter_start_time) / batch_size
            data_time = self.iter_start_time - self.iter_data_time
            info_string = (
                f'=> epoch: {self.curr_ep:5}, '
                f'time: {process_time:.3f}, data: {data_time:.3f}, '
                f'training loss is {loss.item():.5f}'
            )
            print(info_string)
            self.logger.login(info_string)
        self.to_restore['total_step'] += 1

        return loss

    def _val_save(self):
        self.to_restore['curr_ep'] = self.curr_ep

        save_dict = {
            "curr_ep": self.curr_ep,
            'best_metric': self.to_restore['best_metric'],
            'best_ep': self.to_restore['best_ep'],
            'to_restore': self.to_restore,
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'train_data_provider': self.train_data_provider.get_state(),
            'val_data_provider': self.val_data_provider.get_state(),
            'torch_random': torch.get_rng_state(),
            'torch_cuda_state': torch.cuda.get_rng_state()
        }
        self.model_manager.save_iter(
            self.curr_ep, save_dict
        )

    def _create_dict_metric(self):
        ''' create a dictionary of metrics to be reported
        '''
        self.dict_metric = {
            'chamfer_dist': [], 'kld': []
        }

    def _update_dict_metric(self, pred, y, **kwargs):
        ''' update the dictionary of metrics according to 
        the predictions.
        '''
        y_, _ = pred
        ch_loss, _ = pytorch3d.loss.chamfer_distance(
            x=y_, y=y, batch_reduction=None
        )
        ch_loss = ch_loss.data.cpu().numpy().tolist()
        self.dict_metric['chamfer_dist'].extend(ch_loss)

    def _post_process_dict_metric(self):
        list_chamfer = self.dict_metric['chamfer_dist']
        self.dict_metric['mean_chamfer'] = sum(list_chamfer) / len(list_chamfer)

    def _val_update(self, split):
        ''' update to_restore during validation
        '''
        best = self.to_restore['best_metric']
        curr = self.dict_metric['mean_chamfer']

        if best > curr:
            self.to_restore['best_metric'] = curr
            self.to_restore['best_ep'] = self.curr_ep

    def _val_info(self, split):
        ep_string = f'Epoch {self.curr_ep:8d}: '
        best = self.to_restore['best_metric']
        curr = self.dict_metric['mean_chamfer']
        best_ep = self.to_restore['best_ep']
        testa_string = f'Test chamfer {curr:.4f} '
        besta_string = f'Best chamfer {best:4f} at epoch {best_ep:8d}'
        p_string = ep_string + testa_string + besta_string
        print(p_string)
        self.logger.login(p_string)
    
    def _post_train(self):
        if getattr(self.opt.train, 'save_at_train', False):
            self.to_restore['curr_ep'] = self.curr_ep

            save_dict = {
                'to_restore': self.to_restore,
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                'train_data_provider': self.train_data_provider.get_state(),
                # 'val_data_provider': self.val_data_provider.get_state(),
                'val_data_provider': None,
                'torch_random': torch.get_rng_state(),
                'torch_cuda_state': torch.cuda.get_rng_state()
            }
            self.model_manager.save_iter(
                self.curr_ep, save_dict
            )

if __name__ == '__main__':
    parse_args()

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

    # make model
    # model = 
    net = getattr(
        model, options.model.lib
    )(**options.model.params).cuda()
    net = nn.parallel.DistributedDataParallel(
        net, device_ids=[options.gpu],
        find_unused_parameters=False
    )

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

    model_manager = ModelManager(
        options.outf, options.manager, True
    )

    import lib.trainers as trainer
    model_trainer = getattr(
        trainer, options.train.lib
    )(
        logger, tvtdp, net, optimizer, scheduler,
        model_manager
    )
    model_trainer.train(options)

    # make optimizer and scheduler

    # train the network
