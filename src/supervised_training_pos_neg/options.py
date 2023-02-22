''' options for running tain.py script
reference: https://github.com/noahcao/Pixel2Mesh/blob/master/options.py

author:
    Zhangsihao Yang

date: 
    20220314

logs: 
    20220627 update for 'from_basic_blocks' option
'''
import os
import re
import sys

import easydict
import numpy as np
import yaml

options = easydict.EasyDict()

# global
options.outf = 'cls'
options.seed = 0

# logger
options.logger = easydict.EasyDict()

# manager
options.manager = easydict.EasyDict()

# dataset
options.data = easydict.EasyDict()

options.data.train = easydict.EasyDict()
options.data.train.dataset = easydict.EasyDict()
options.data.train.dataset.params = easydict.EasyDict()
options.data.train.dataloader = easydict.EasyDict()

options.data.val = easydict.EasyDict()
options.data.val.dataset = easydict.EasyDict()
options.data.val.dataset.params = easydict.EasyDict()
options.data.val.dataloader = easydict.EasyDict()

options.data.test = easydict.EasyDict()
options.data.test.dataset = easydict.EasyDict()
options.data.test.dataset.params = easydict.EasyDict()
options.data.test.dataloader = easydict.EasyDict()

# model
options.model = easydict.EasyDict()
options.model.params = easydict.EasyDict()
options.model.params.base_encoder = \
easydict.EasyDict()

# optimization
options.optim = easydict.EasyDict()
options.optim.params = easydict.EasyDict()
options.optim.scheduler = easydict.EasyDict()
options.optim.scheduler.params = easydict.EasyDict()

# validation and test
options.val = easydict.EasyDict()
options.test = easydict.EasyDict()

# train
options.train = easydict.EasyDict()
options.train.loss_params = easydict.EasyDict()
options.train.params_0 = easydict.EasyDict()
options.train.params_1 = easydict.EasyDict()
options.train.params_2 = easydict.EasyDict()


def _update_dict(full_key: str, val, d):
    """ Update dictionary given a value. The dictionary is 
    a global variable. If the key is not in the dictionary, 
    it will be added.

    Args:
        full_key (str): The position of the key.
        val (dict): The values used to update the dictionary.
        d (EasyDict): The dictionary to be updated.
    """
    for vk, vv in val.items():
        if vk == 'from_basic_blocks':
            # load the file
            options_file = os.path.join(os.getcwd(), 'exps', 'basic_blocks', vv)
            options_dict = _safe_load(options_file)
            # update the dict with the loaded dict
            _update_dict(full_key, options_dict, d)
        if isinstance(vv, list):  # The value of the key is list.
            d[vk] = np.array(vv)  # Store it as a numpy array.
        elif isinstance(vv, dict):  # The value of the key is dictionary.
            # create a easydict automatically
            if vk not in d:
                d[vk] = easydict.EasyDict()
            _update_dict(full_key + "." + vk, vv, d[vk])
        else:  # At the leaf of the dictionary.
            d[vk] = vv


def _safe_load(options_file: str):
    with open(options_file, 'r', encoding='utf-8') as yfile:
        options_dict = yaml.safe_load(yfile)
    if options_dict is None:
        return {}
    return options_dict


def _update_options(options_file):
    if not os.path.exists(options_file):
        sys.exit(
            f'file {options_file} '
            'does not exist'
        )

    options_dict = _safe_load(
        options_file
    )
    if 'based_on' in options_dict:
        base_folder = os.path.dirname(
            options_file
        )
        for base in options_dict['based_on']:
            base_options_file = os.path.join(base_folder, base)
            _update_options(base_options_file)
        options_dict.pop('based_on')
    _update_dict('', options_dict, options)


def update_options(options_file):
    _update_options(options_file)
