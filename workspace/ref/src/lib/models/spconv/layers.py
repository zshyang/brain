''' layers for sparse convolution

function:
    f00 sparse_linear(in_channels, out_channels):
    f01 single_conv(in_channels, out_channels, kernel_size, stride, indice_key=None)

author:
    zhangsihao yang

date:
    20220822

logs:
    20220822
        created
        f00
        f01
'''
import torch.nn as nn

import spconv


def sparse_linear(in_channels, out_channels):
    return spconv.SparseSequential(
        nn.Linear(in_channels, out_channels),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def single_conv(
    in_channels, out_channels, kernel_size, stride, indice_key=None
):
    ''' single convolution layer

    args:
        in_channels
        out_channels
        kernel_size
        indice_key

    returns:
        a sparse sequence
    '''
    if stride == 1:
        return spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, out_channels, kernel_size, 
                bias=True, indice_key=indice_key
            ),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
    else:
        return spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, out_channels, kernel_size, 
                bias=True, indice_key=indice_key
            ),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SparseMaxPool3d(3, stride)
        )
