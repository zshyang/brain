''' train script.
Author: Zhangsihao Yang
Date: 2022-0314

tvtdp = train validation test data provider
'''
from __future__ import print_function

import argparse
import os
import random
import sys
import time

# from lib.options import options, update_options


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--outf', type=str, 
        default='occo/pretrain/p/'
    )

    args = parser.parse_args()
    return args


if __name__  == '__main__':
    args = parse_args()
    exit_file = os.path.join(
        '../runtime', args.outf, 'cont.txt'
    )
    if os.path.exists(exit_file):
        sys.exit(124)
    # os.system(f'rm {exit_file}')
