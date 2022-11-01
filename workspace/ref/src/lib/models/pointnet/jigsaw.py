''' network for pointnet jigsaw

author:
    Zhangsihao Yang

date:
    04/23/2022

name convention:
    bs = batch size
    k = class
    tm = translation matrix
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.pointnet.util import PointNetFeat


class SimpleNet(nn.Module):
    def __init__(
        self, num_channel=3,
        **kwargs
    ):
        super(
            SimpleNet, self
        ).__init__()

        self.feat = PointNetFeat(
            global_feat=False,
            feature_transform=False,
            channel=num_channel,
            **kwargs
        )

    def forward(self, x):
        # x = x.transpose(1, 2)
        bs, _, num_points = x.size()

        x, tm, tm1, y = self.feat(x)
        # tm1 : [b, 64, 64]

        return y

class Net(nn.Module):
    def __init__(
        self, k, num_channel=3,
        **kwargs
    ):
        super(Net, self).__init__()
        self.k = k

        self.feat = PointNetFeat(
            global_feat=False,
            feature_transform=True,
            channel=num_channel
        )
        self.conv1 = nn.Conv1d(
            1088, 512, 1
        )
        self.conv2 = nn.Conv1d(
            512, 256, 1
        )
        self.conv3 = nn.Conv1d(
            256, 128, 1
        )
        self.conv4 = nn.Conv1d(
            128, k, 1
        )
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        bs, _, num_points = x.size()

        x, tm, tm1, y = self.feat(x)
        # tm1 : [b, 64, 64]

        x = F.relu(
            self.bn1(self.conv1(x))
        )
        x = F.relu(
            self.bn2(self.conv2(x))
        )
        x = F.relu(
            self.bn3(self.conv3(x))
        )
        x = self.conv4(x)
        x = x.transpose(
            2, 1
        ).contiguous()
        x = F.log_softmax(
            x.view(-1, self.k), dim=-1
        )

        return x, tm1, y
