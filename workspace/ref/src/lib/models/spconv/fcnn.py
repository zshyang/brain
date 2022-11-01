''' fully convolution neural network

class:
    c00 Network
        00  __init__(self)
        01  forward(self, x)
            <-- f00

function:
    f00 sparse_dict_to_sparse_tensor(sparse_dict)

author:
    zhangsihao yang

date:
    20220822

logs:
    20220822
        file    created
        c00-00
        c00-01
        f00     finished
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.spconv.layers import single_conv, sparse_linear

import spconv


def sparse_dict_to_sparse_tensor(sparse_dict):
    sparse_tensor = spconv.SparseConvTensor(
        sparse_dict['features'], sparse_dict['indices'],
        sparse_dict['spshape'], sparse_dict['batch_size']
    )
    return sparse_tensor


class Network(nn.Module):
    def __init__(self, **kwargs):
        super(Network, self).__init__()

        embedding_channel = 1024
        channels = (32, 48, 64, 96, 128)
        num_class = 30

        self.mlp1 = sparse_linear(3, channels[0])
        self.conv1 = single_conv(channels[0], channels[1], 3, 1)
        self.conv2 = single_conv(channels[1], channels[2], 3, 2)
        self.conv3 = single_conv(channels[2], channels[3], 3, 2)
        self.conv4 = single_conv(channels[3], channels[4], 3, 2)
        self.conv5 = spconv.SparseSequential(
            single_conv(
                channels[4],
                embedding_channel // 4, 3, 2
            ),
            single_conv(
                embedding_channel // 4,
                embedding_channel // 2, 3, 2
            ),
            single_conv(
                embedding_channel // 2,
                embedding_channel, 3, 2
            ),
        )

        self.global_max_pool = spconv.SparseSequential(
            spconv.SparseMaxPool3d(3)
        )

        self.mlp = nn.Sequential(
            nn.Linear(embedding_channel, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        # grid = get_grid_coords(x)

        y = sparse_dict_to_sparse_tensor(x)

        y = self.mlp1(y)

        y = self.conv1(y)
        # test(grid, y, x)
        # y1 = y.dense()

        y = self.conv2(y)
        # y2 = y.dense()
        y = self.conv3(y)
        # y3 = y.dense()
        y = self.conv4(y)
        # y4 = y.dense()

        # y = interpolate_features(grid, [y1, y2, y3, y4])

        y = self.conv5(y)

        # x = ME.cat(x1, x2, x3, x4)
        # y = self.conv5(x.sparse())

        # x1 = self.global_max_pool(y)
        # x2 = self.global_avg_pool(y)

        x1 = self.global_max_pool(y)

        x1 = x1.dense().view(x['batch_size'], -1)

        return self.mlp(x1)
