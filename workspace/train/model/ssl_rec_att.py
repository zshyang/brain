'''
author:
    zhangsihao yang

date:
    20220910

logs:
    20220910
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
        self.decoder = PointNetDecoder(**de_config)

    def forward(
        self, verts, faces, centers, 
        normals, ring_1, ring_2, ring_3, **kwargs
    ):
        fea = self.encoder(verts, faces, centers, normals, ring_1, ring_2, ring_3)
        y = self.decoder(fea)
        return y, fea
