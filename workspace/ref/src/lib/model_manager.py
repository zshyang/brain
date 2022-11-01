''' model manager. store checkpoint produced from running.
reference from: https://github.com/facebookresearch/dino/blob/main/utils.py
Author: Zhangsihao Yang
Date: 2022-0314

cf = checkpoint filename
nd = net dict
pd = pretrained dict
'''
import os

import torch
from lib.utils import is_main_process


class ModelManager():
    def __init__(self, root, opt, verbose=False):
        root = os.path.join('/runtime', root)

        self.root = root
        self.prefix = opt.ckpt_prefix
        self.verbose = verbose

        os.makedirs(self.root, exist_ok=True)

    def _search_file(self, file_path)->bool:
        if not os.path.isfile(file_path):
            if self.verbose:
                print(
                    'The specified '
                    'iteration is not '
                    'found'
                )
                print('Train from scratch')
            return False
        if self.verbose:
            print(f'Found checkpoint at {file_path}')
        return True

    def _load_dict(self, checkpoint, **kwargs):

        for key, value in kwargs.items():
            if key == 'torch_random':
                torch.set_rng_state(checkpoint[key])
                print('=> set torch random state')
            if key == 'torch_random':
                torch.set_rng_state(checkpoint[key])
                print('=> set torch random state')
            if key in checkpoint and value is not None:
                try:
                    msg = value.load_state_dict(
                        checkpoint[key], strict=True)
                    print(f'=> loaded {key} with msg {msg}')
                except TypeError:
                    try:
                        msg = value.load_state_dict(
                            checkpoint[key])
                        print(f'=> loaded {key}')
                    except ValueError:
                        print(f'=> failed to load {key}')
                except AttributeError:
                    value.set_state(checkpoint[key])
                    print(f'=> loaded {key}')
            
            else:
                print(f'=> key {key} not found')

    def load_ep(self, ep, run_variables=None, **kwargs):
        ''' load the save state from certain epoch.
        '''

        ckpt_path = os.path.join(
            self.root, 
            f'{self.prefix}_{ep}.pth'
        )
        if not self._search_file(
            ckpt_path    
        ):
            return

        # open checkpoint file
        checkpoint = torch.load(
            ckpt_path, map_location='cpu'
        )

        # key is what to look for in the checkpoint file
        # value is the object to load
        # example: {'state_dict': model}
        self._load_dict(checkpoint, **kwargs)

        # reload variable important for the run
        if run_variables is not None:
            for var_name in run_variables:
                run_variables[var_name] = \
                    checkpoint['to_restore'][var_name]
        print(f'=> loaded to_restore')

    def _copy_state(self, net, ckpt):
        '''
        ref: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3

        Args:
            ckpt: must contain key net
        '''
        nd = net.state_dict()
        pd = ckpt['net']
        pd = {
            k: v for k, v in pd.items() if \
            k in nd and pd[k].size() == nd[k].size()
        }

        if self.verbose:
            print('=' * 27)
            print('Restored Params and Shapes:')
            for k, v in pd.items():
                print(k, ': ', v.size())
            print('=' * 68)

        nd.update(pd)
        net.load_state_dict(nd)

    def load_pretrain(self, net, cf):
        ckpt_path = os.path.join(self.root, cf)
        if not self._search_file(ckpt_path):
            return
        ckpt = torch.load(ckpt_path)
        self._copy_state(net, ckpt)

    def save_iter(self, iteration, save_dict, **kwargs):
        '''
        **kwargs is only for extra parameter required by
        torch.save function.
        '''
        ckpt_path = os.path.join(
            self.root, f'{self.prefix}_{iteration}.pth'
        )
        if is_main_process():
            torch.save(save_dict, ckpt_path, **kwargs)
