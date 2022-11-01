'''
author:
    zhangsihao yang

logs:
    20220919
        file created
'''
import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, **kwargs):
        super(Network, self).__init__()
        self.structure_scale = kwargs.get('structure_scale', 1.0)

        self.bn0 = nn.BatchNorm1d(2048)
        out_channel0 = int(2048 * self.structure_scale)
        self.linear0 = nn.Linear(2048, out_channel0)
        self.bn1 = nn.BatchNorm1d(out_channel0)

        out_channel1 = int(4096 * self.structure_scale)
        self.linear1 = nn.Linear(out_channel0, out_channel1)
        self.bn2 = nn.BatchNorm1d(out_channel1)

        out_channel2 = int(1024 * self.structure_scale)
        self.linear2 = nn.Linear(out_channel1, out_channel2)
        self.bn3 = nn.BatchNorm1d(out_channel2)
        # change to layer normal, and instance norm

        self.linear3 = nn.Linear(out_channel2, 50)
        # add drop out 0.1

        self.elu = nn.ReLU()

    def forward(self, x):
        # x = self.elu(x)
        x = self.bn0(x)
        x = self.bn1(self.linear0(x))
        x = self.elu(x)
        x = self.bn2(self.linear1(x))
        x = self.elu(x)
        x = self.bn3(self.linear2(x))
        x = self.elu(x)
        x = self.linear3(x)
        return x
