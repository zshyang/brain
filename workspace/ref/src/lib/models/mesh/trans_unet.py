''' mesh model start with transformer, followed by sparse unet, and end with a maxpooling

name convention:


class:
    

function:


author:
    Zhangsihao Yang

date:
    20220605

logs:
    20220605
        file created
'''
import spconv
import torch.nn as nn
from lib.models.mesh.mesh_transformer import BlockTransformerEncoder
from lib.models.spconv.unet import Network as unet


class TwoLayerMLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden, **kwargs):
        super(TwoLayerMLP, self).__init__()

        self.fc1 = nn.Linear(in_channels, hidden)
        self.fc2 = nn.Linear(hidden, out_channels)
        # self.bn1 = nn.BatchNorm1d(hidden, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, features):
        # y = self.relu(self.bn1(self.fc1(features)))
        y = self.relu(self.fc1(features))
        y = self.fc2(y)
        return y


class Network(nn.Module):
    def __init__(self, bte_kwargs={}, unet_kwargs={}, mlp_kwargs={}, **kwargs):
        '''
        args:
            bte_kwargs: dictionary

        name convention:
            bte = block transformer encoder
        '''
        super(Network, self).__init__()

        self.block_transformer_encoder = BlockTransformerEncoder(**bte_kwargs)
        self.spconv_unet = unet(**unet_kwargs)
        self.maxpool = spconv.SparseMaxPool3d(unet_kwargs['nb'])
        self.mlp = TwoLayerMLP(**mlp_kwargs)

    def forward(self, vs, vms, fs, fms, spidx, spshape, batch_size, **kwargs):
        bte_input_kwargs = {'vs': vs, 'vms': vms, 'fs': fs, 'fms': fms}
        sparse_block_features = self.block_transformer_encoder(**bte_input_kwargs)
        unet_sparse_input = spconv.SparseConvTensor(
            sparse_block_features, spidx, spshape, batch_size
        )
        unet_out = self.spconv_unet(unet_sparse_input)
        unet_max = self.maxpool(unet_out)
        unet_max_feature = unet_max.features
        y = self.mlp(unet_max_feature)
        return y
