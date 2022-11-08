''' the network with meshnet's encoder and adding with a classifier.

author:
    zhangsihao yang

logs:
    20221106
        file    created
'''
import torch
import torch.nn as nn
from model.meshnet2_attention import NetworkBeforeLinear
from model.pnvae import PointNetDecoder


class Network(nn.Module):
    def __init__(self, en_config={}, de_config={}, **kwargs):
        super(Network, self).__init__()

        self.encoder = NetworkBeforeLinear(**en_config)

        in_channel = de_config['in_channel']
        num_cls = de_config['num_cls']
        self.classifier = nn.Sequential(
            nn.Linear(in_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_cls)
        )

    def forward(
        self, verts, faces, centers, 
        normals, ring_1, ring_2, ring_3, **kwargs
    ):
        fea = self.encoder(verts, faces, centers, normals, ring_1, ring_2, ring_3)
        y = self.classifier(fea)
        return y, fea
