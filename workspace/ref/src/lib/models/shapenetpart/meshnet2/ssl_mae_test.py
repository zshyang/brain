'''
author:
    zhangsihao yang

logs:
    20220918
        file created
'''
import math

import torch
import torch.nn as nn
from lib.models.mesh.basic_transformer import (TransformerEncoder,
                                               TransformerEncoderLayer)
from lib.models.meshnet.layers import MaxPoolFaceFeature
from lib.models.meshnet.spatial_descriptor import PointDescriptor
from lib.models.meshnet.structural_descriptor import (ConvSurface,
                                                      NormalDescriptor)
from lib.models.modelnet40.meshnet2.meshnet2_attention import MeshBlock
from lib.models.modelnet40.meshnet2.pnvae import PointNetDecoder
from lib.models.modelnet40.meshnet2.ssl_mae import MaskedMeshNetPerFaceFea
from torch.nn.parameter import Parameter


class Network(nn.Module):
    def __init__(self, en_config={}, de_config={}, **kwargs):
        super(Network, self).__init__()

        self.encoder = MaskedMeshNetPerFaceFea(**en_config)
        self.decoder = PointNetDecoder(**de_config)

    def forward(
        self, verts, faces, centers, normals, ring_1, ring_2, ring_3, **kwargs
    ):
        fea = self.encoder(verts, faces, centers, normals, ring_1, ring_2, ring_3)
        # y = self.decoder(fea)
        return fea
