'''
Zhangsihao Yang
04/14/2022

bc = block concat
bd = bottleneck dimension
be = block embedding
bs = batch size
bex = block extraction
bte = block transformer encoder
c3e = convolution 3d encoder
e = encoder
ed = end dimension
ee = empty embedding
el = encoder layer
fd = final dimension
ffv = face fetch vertex
hd = hidden dimension
ic = in channel
m = mask
me = mesh encoder
msl = maximum sequence length
nh = number of head
nl = number of layers
pe = position embedding
pcd = point cloud decoder
pel = position embedding layer
se = stopping_embeddings
svc = stopping vertex concat
t = transformer
ve = vertex embedding

'''
import math

import torch
import torch.nn as nn
from lib.model.mesh.basic_transformer import (TransformerEncoder,
                                                 TransformerEncoderLayer)
from torch.nn.parameter import Parameter


class VertexEmbedding(nn.Module):
    def __init__(self, hd):
        super(
            VertexEmbedding, self
        ).__init__()
        self.conv1 = nn.Conv1d(3, 32, 1)
        self.conv2 = nn.Conv1d(32, hd, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)

    def forward(self, vs):
        vs = vs.transpose(2, 1)
        x = self.relu(
            self.bn1(self.conv1(vs))
        )
        x = self.conv2(x)
        x = x.transpose(2, 1)
        return x


class StopEmbedding(nn.Module):
    ''' the embedding for stop signs in 
    face generation.
    '''
    def __init__(self, embedding_dim):
        super(StopEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        # Initialize the tensor
        # https://pytorch.org/docs/stable/nn.init.html
        self.stopping_embeddings = Parameter(
            nn.init.xavier_uniform_(
                torch.empty(
                    1, 2, self.embedding_dim
                )
            )
        )

    def forward(self, batch_size):
        return self.stopping_embeddings.repeat(
            (batch_size, 1, 1)
        )


class StoppingVertexConcat(nn.Module):
    def __init__(self, hd):
        super(
            StoppingVertexConcat, self
        ).__init__()
        self.se = StopEmbedding(hd)

    def forward(self, x):
        se = self.se(x.shape[0])
        x = torch.cat((se, x), 1)
        return x


class FaceFetchVertex(nn.Module):
    def __init__(self):
        super(
            FaceFetchVertex, self
        ).__init__()

    def forward(self, fs, x):
        # face value embeddings are gathered 
        # from vertex embeddings.

        # reshape and repeat faces for gather 
        # function.

        # fs : [b, nf]
        fs = fs.unsqueeze(-1)
        # fs : [b, nf, 1]

        # repeat the last dimension as the 
        # vertex_embeddings
        # fs : [b, nf, 1]
        fs = fs.repeat((1, 1, x.shape[-1]))
        # fs : [b, nf, 128]

        # gather the face_embeddings.
        # the usage of torch.gather:
        # https://pytorch.org/docs/stable/generated/torch.gather.html
        # x : [b, nv, 128]
        x = torch.gather(
            input=x, dim=1, 
            index=fs.type(torch.int64),
        )
        # y : [b, nf, 128]
        return x


class BlockEmbedding(nn.Module):
    """The embedding for block representation.
    """
    def __init__(self, embedding_dim):
        super(
            BlockEmbedding, self
        ).__init__()
        self.embedding_dim = embedding_dim
        # Initialize the tensor
        # https://pytorch.org/docs/stable/nn.init.html
        self.block_embedding = Parameter(
            nn.init.xavier_uniform_(
                torch.empty(
                    1, 1, self.embedding_dim
                )
            )
        )

    def forward(self, batch_size):
        # tile usage:
        # https://pytorch.org/docs/stable/generated/torch.tile.html
        return self.block_embedding.repeat(
            (batch_size, 1, 1)
        )


class BlockConcat(nn.Module):
    def __init__(self, hd):
        super(BlockConcat, self).__init__()
        self.be = BlockEmbedding(hd)
        self.pl = torch.nn.ConstantPad1d(
            (1, 0), 0.0
        )

    def forward(self, x, fms):
        be = self.be(x.shape[0])
        x = torch.cat((be, x), dim=1)
        fms = self.pl(fms)
        return x, fms
        

class GeneralTransformer(nn.Module):
    def __init__(
        self, msl, hd, nh, nl
    ):
        super(
            GeneralTransformer, self
        ).__init__()
        # position embedding layer
        position = torch.arange(msl)
        self.pel = nn.Embedding(
            num_embeddings=msl,
            embedding_dim=hd
        )
        self.register_buffer(
            'position', position
        )
        # The network.
        el = TransformerEncoderLayer(
            hd, nh,
            batch_first=True,
        )
        self.e = TransformerEncoder(
            el, nl
        )


    def forward(self, x, m):
        pe = self.pel(
            self.position[:x.shape[1]]
        )
        # Aggregate embeddings.
        # None is to expand the dimension.
        x = x + pe[None]
        x = self.e(
            src=x,
            src_key_padding_mask=m
        )
        return x


class BlockExtraction(nn.Module):
    def __init__(self):
        super(
            BlockExtraction, self
        ).__init__()

    def forward(self, x):
        x = x[:, 0]
        return x


class BlockTransformerEncoder(nn.Module):
    def __init__(
        self, msl, hd, nh, nl, ed
    ):
        '''
        name convention:
            ed = end dimension
            hd = hidden dimension
            nh = number of heads
            nl = number of layers
            ve = vertex embedding
            msl = maximum sequence length
        '''
        super(
            BlockTransformerEncoder, self
        ).__init__()
        self.ve = VertexEmbedding(hd)
        self.svc = StoppingVertexConcat(hd)
        self.ffv = FaceFetchVertex()
        self.bc = BlockConcat(hd)
        self.t = GeneralTransformer(
            msl, hd, nh, nl
        )
        self.bex = BlockExtraction()
        self.end_linear = nn.Linear(
            hd, ed
        )

    def forward(self, vs, vms, fs, fms, **kwargs):
        x = self.ve(vs)
        x = self.svc(x)
        x = self.ffv(fs, x)
        x, fms = self.bc(x, fms)
        x = self.t(x, fms.type(torch.bool))
        x = self.bex(x)
        x = self.end_linear(x)
        return x


class MeshEncoder(nn.Module):
    def __init__(
        self, msl, hd, nh, nl, ed, bd, nb,
        **kwargs
    ):
        super(MeshEncoder, self).__init__()
        self.bte = BlockTransformerEncoder(
            msl, hd, nh, nl, ed
        )
        self.bs = BlockShaper(ed, nb)
        self.c3e = Conv3dEncoder(
            bd, in_channels=ed, dim=nb
        )
    
    def forward(
        self, vs, vms, fs, fms, gi
    ):
        x = self.bte(vs, vms, fs, fms)
        x = self.bs(x, gi)
        x = self.c3e(x)
        return x


class EmptyEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(
            EmptyEmbedding, self
        ).__init__()

        self.embed_dim = embed_dim

        self.empty_embedding = Parameter(
            nn.init.xavier_uniform_(
                torch.empty(
                    1, self.embed_dim
                )
            )
        )

    def forward(self):
        return self.empty_embedding


class BlockShaper(nn.Module):
    def __init__(self, ed, nb, **kwargs):
        super(
            BlockShaper, self
        ).__init__()
        self.ed = ed
        self.nb = nb
        self.ee = EmptyEmbedding(ed)

    def forward(self, x, gi):
        nb = self.nb
        ee = self.ee()
        x = torch.cat((ee, x), 0)
        x = x.unsqueeze(0)
        batch_size = gi.shape[0]
        x = x.repeat((batch_size, 1, 1))

        gi = gi.unsqueeze(-1)
        gi = gi.repeat(
            (1, 1, self.ed)
        )
        x = torch.gather(
            input=x, dim=1, 
            index=gi
        )
        x = x.reshape(
            batch_size, nb, nb, nb, -1
        )
        return x


class Conv3dEncoder(nn.Module):
    def __init__(
        self, bd, in_channels=3, dim=32,
        out_conv_channels=512):
        super(
            Conv3dEncoder, self
        ).__init__()
        conv1_channels = int(
            out_conv_channels / 8
        )
        conv2_channels = int(
            out_conv_channels / 4
        )
        conv3_channels = int(
            out_conv_channels / 2
        )
        self.out_conv_channels = \
            out_conv_channels
        self.out_dim = int(dim / 16)

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels, 
                out_channels=conv1_channels,
                kernel_size=4,
                stride=2, padding=1, 
                bias=False
            ),
            nn.BatchNorm3d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv1_channels,
                out_channels=conv2_channels,
                kernel_size=4,
                stride=2, padding=1, 
                bias=False
            ),
            nn.BatchNorm3d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv2_channels,
                out_channels=conv3_channels,
                kernel_size=4,
                stride=2, padding=1, 
                bias=False
            ),
            nn.BatchNorm3d(conv3_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv3_channels,
                out_channels=out_conv_channels,
                kernel_size=4,
                stride=2, padding=1,
                bias=False
            ),
            nn.BatchNorm3d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fd = out_conv_channels * \
            self.out_dim * \
            self.out_dim * \
            self.out_dim
        self.out = nn.Sequential(
            nn.Linear(self.fd, bd)
        )

    def forward(self, x):
        bs = x.shape[0]
        ic = x.shape[-1]
        nb = x.shape[1]
        x = x.reshape(bs, -1, ic)
        x = x.transpose(1, 2)
        x = x.reshape(bs, ic, nb, nb, nb)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, self.fd)
        x = self.out(x)
        return x


class PointCloudDecoder(nn.Module):
    def __init__(self, bd, **kwargs):
        super(
            PointCloudDecoder, self
        ).__init__()
        self.linear0 = nn.Linear(bd, 1024)
        self.linear1 = nn.Linear(
            1024, 6144)
        self.elu = nn.ELU()

    def forward(self, x):
        batch_size = x.shape[0]
        point_cloud = self.elu(
            self.linear0(x)
        )
        point_cloud = self.linear1(
            point_cloud
        )
        point_cloud = point_cloud.reshape(
            batch_size, -1, 3
        )
        return point_cloud


class MeshAE(nn.Module):
    def __init__(self, **kwargs):
        super(MeshAE, self).__init__()
        self.me = MeshEncoder(**kwargs)
        self.pcd = PointCloudDecoder(
            **kwargs
        )

    def forward(
        self, vs, vms, fs, fms, gi
    ):
        x = self.me(vs, vms, fs, fms, gi)
        x = self.pcd(x)
        return x
