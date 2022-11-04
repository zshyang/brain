''' test the pretrain autoencoder

author:
    zhangsihao yang

logs:
    20221101    file created
'''
import torch
import argparse
from options import update_options

def parse_args():
    parser = argparse.ArgumentParser()

    str_help = 'experiment options file name'
    parser.add_argument('--options',
        help=str_help, required=True, type=str
    )

    args = parser.parse_args()

    update_options(args.options)


def test():
    # visit the dataset and get the features 

    # partation the features 10 times to get the accuracy

    pass


if __name__ is '__main__':
    parse_args()
    # set up the dataset and dataloader

    # set up the network 
    # load the pretrained weights

    # do the test
    test()
