'''
Zhangsihao Yang
04/14/2022

lp = left partion
tl = time left
tpt = total pass time

'''
from time import time

from lib.dataset.mesh_util import process_flatten_meshes
from lib.trainer.trainer import ModelTrainer


class MeshAETrainer(ModelTrainer):
    def _prepare_train(self):
        self.to_restore = {
            'curr_ep': -1, 'ts': 0, 
            'bl': 9999, 'ble': -1,
        }

        if self.train_opt.mode == 2:
            self._load_net()

        self.start_ep = self.to_restore['curr_ep'] + 1
        self.end_ep = self.train_opt.max_epoch + 1
    
    def _criterion(self, pred, y, **kwargs):
        ''' compute the reconstruction loss 
        '''
        # compute chamfer distance
        import pytorch3d.loss
        loss, _ = \
            pytorch3d.loss.chamfer_distance(
            x=pred, y=y
        )

        if self.to_restore['ts'] % \
            self.train_opt.train_freq == 0:
            print(
                f'=> training loss '
                f'is {loss:.9f}'
            )

            tpt = int(
                time() - self.st
            )
            lp = (
                (
                    self.to_restore['ts'] % \
                    len(self.train_data_provider)
                ) + 1
            ) / len(self.train_data_provider)
            tl = int(
                tpt * (1. / lp - 1.)
            )
            print(
                f'time pass {tpt} '
                f'time left {tl}'
            )

        self.to_restore['ts'] += 1
        
        return loss

    def _create_dict_metric(self):
        ''' create a dictionary of metrics 
        to be reported
        '''
        self.dict_metric = {
            'chamfer_dist': []
        }

    def _update_dict_metric(
        self, pred, y
    ):
        ''' update the dictionary of 
        metrics according to the 
        predictions.
        '''
        import pytorch3d.loss
        ch_loss, _ = \
        pytorch3d.loss.chamfer_distance(
            x=pred, y=y,
            batch_reduction=None
        )

        ch_loss = ch_loss.data.cpu()
        ch_loss = ch_loss.numpy().tolist()
        self.dict_metric[
            'chamfer_dist'
        ].extend(ch_loss)

    def _post_process_dict_metric(
        self, pred, **kwargs
    ):
        list_chamfer = self.dict_metric[
            'chamfer_dist'
        ]

        self.dict_metric['mean_chamfer'] = \
            sum(list_chamfer) / \
            len(list_chamfer)

        # save mesh and point cloud
        pass

        # generate html files



    def _val_update(self, split):
        ''' update to_restore during validation
        '''
        best = self.to_restore['bl']
        curr = self.dict_metric[
            'mean_chamfer'
        ]
        if  best > curr:
            self.to_restore[
                'bl'
            ] = curr
            self.to_restore[
                'ble'
            ] = self.curr_ep

    def _val_info(self, split):
        ep_string = f'Epoch {self.curr_ep:8d}: '
        best = self.to_restore['bl']
        curr = self.dict_metric['mean_chamfer']
        best_ep = self.to_restore['ble']
        testa_string = f'Test chamfer {curr:.4f} '
        besta_string = f'Best chamfer {best:4f} at epoch {best_ep:8d}'
        p_string = ep_string + testa_string + besta_string
        print(p_string)
        self.logger.login(p_string)

    def _val_save(self):
        self.to_restore['curr_ep'] = \
            self.curr_ep
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
