''' Use the features processed by meshnet 
and pass those features into transformer.

Whether or not to sort the input faces is underdetermine.
Get results first!

class:
    c00 Network(nn.Module)
        00 __init__(self, num_faces, cfg, num_cls, **kwargs)
        01 forward(self, verts, faces, centers, normals, ring_1, ring_2, ring_3, **kwargs)

    c01 GeneralTransformerNoPE(nn.Module)
        00 __init__(self, msl, hd, nh, nl)
        01 forward(self, x)
    
    c02 ConvFace(nn.Module)
        00 __init__(self, in_channel, out_channel, num_neighbor)
            <-- c05
        01 forward(self, fea, ring_n, pool_idx)

    c03 ConvFaceBlock(nn.Module)
        00 __init__(self, in_channel, growth_factor, num_neighbor)
        01 forward(self, fea, ring_n, pool_idx)

    c04 MeshBlock(nn.ModuleDict)
        00 __init__(self, in_channel, num_neighbor, num_block, growth_factor)
        01 forward(self, fea, ring_n, pool_idx=None)
    
    c05 Linear4D(nn.Module)
        00 __init__(self, in_channel, out_channel)
        01 forward(self, x)
    
    c06 Embedder

author:
    zhangsihao yang

date:
    20220825

logs:
    20220825
        file    created
    20220830
        c02     created
        c03     created
        c04     created
    20220902
        c05     created
    20220905
        c06     created
'''
import torch
import torch.nn as nn
from lib.models.mesh.basic_transformer import (TransformerEncoder,
                                              TransformerEncoderLayer)
from lib.models.meshnet.layers import MaxPoolFaceFeature
from lib.models.meshnet.spatial_descriptor import PointDescriptor
from lib.models.meshnet.structural_descriptor import (ConvSurface,
                                                     NormalDescriptor)
import math


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class GeneralTransformerNoPE(nn.Module):
    def __init__(
        self, msl, hd, nh, nl
    ):
        '''
        msl = maximum sequence length
        hd = hidden dimension
        nh = number of head
        nl = number of layers
        '''

        super(
            GeneralTransformerNoPE, self
        ).__init__()
        # position embedding layer
        # position = torch.arange(msl)
        # self.pel = nn.Embedding(
        #     num_embeddings=msl,
        #     embedding_dim=hd
        # )
        # self.register_buffer(
        #     'position', position
        # )
        # The network.
        el = TransformerEncoderLayer(
            hd, nh,
            batch_first=True,
        )
        self.e = TransformerEncoder(
            el, nl
        )


    def forward(self, x):
        # pe = self.pel(
        #     self.position[:x.shape[1]]
        # )
        # # Aggregate embeddings.
        # # None is to expand the dimension.
        # x = x + pe[None]
        x = self.e(
            src=x,
            # src_key_padding_mask=m
        )
        return x


class Linear4D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Linear4D, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel)
    
    def forward(self, x):
        num_meshes, in_channel, num_faces, num_neigh = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, in_channel)
        y = self.linear(x)
        y = y.view(num_meshes, num_faces, num_neigh, -1)
        y = y.permute(0, 3, 1, 2)
        return y


class ConvFace(nn.Module):
    def __init__(
        self, in_channel, out_channel, num_neighbor, use_neighbor=True,
        op_method='sum', gather_range='local', gather_method='softmax', **kwargs
    ):
        """
        Args:
            in_channel: number of channels in feature

            out_channel: number of channels produced by convolution

            num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(ConvFace, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_neighbor = num_neighbor
        self.use_neighbor = use_neighbor
        self.op_method = op_method
        self.gather_range = gather_range
        self.gather_method = gather_method

        if self.use_neighbor:
            self.concat_mlp = nn.Sequential(
                nn.Conv1d(self.in_channel, self.out_channel, 1),
                nn.BatchNorm1d(self.out_channel),
                nn.ReLU(),
            )

            if self.op_method == 'attention':
                self.key_linear = Linear4D(in_channel, 128)
                self.query_linear = Linear4D(in_channel, 128)

            if self.gather_range == 'local':
                self.softmax = nn.Softmax(dim=3)
            elif self.gather_range == 'global':
                self.softmax = nn.Softmax(dim=2)
            else:
                raise TypeError('not valid gather range')

        else:
            self.concat_mlp = nn.Sequential(
                nn.Conv1d(self.in_channel, self.out_channel, 1),
                nn.BatchNorm1d(self.out_channel),
                nn.ReLU(),
            )

            self.key_linear = Linear4D(66, 128)
            self.query_linear = Linear4D(66, 128)

            self.softmax = nn.Softmax(dim=2)

    def forward(self, fea, ring_n, pool_idx, pos_embed):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor]

            pool_idx: indices of faces to be considered for spatial pooling
            [num_faces]//2 OR [num_faces]//4

        Returns:
            conv_fea: features produced by convolution of faces with its
            n-Ring neighborhood features
            [num_meshes, out_channel, num_faces]
        """
        num_meshes, num_channels, _ = fea.size()
        _, num_faces, _ = ring_n.size()       

        ''' Gather features at face neighbors only at pool_idx '''
        if self.use_neighbor:
            if self.gather_range == 'local':
                # fea [num_meshes, in_channel, num_faces]
                fea = fea.unsqueeze(3)
                # fea [num_meshes, in_channel, num_faces, 1]

                # ring_n [num_meshes, num_faces//2, 6]
                ring_n = ring_n.unsqueeze(1)
                # ring_n [num_meshes, 1, num_faces//2, 6]
                ring_n = ring_n.expand(num_meshes, num_channels, num_faces, -1)
                # [num_meshes, in_channel, num_faces//2, 6]

                neighbor_fea = fea[
                    torch.arange(num_meshes)[:, None, None, None],
                    torch.arange(num_channels)[None, :, None, None],
                    ring_n
                ]
                # neighbor_fea [num_meshes, in_channel, num_faces//2, 6, 1]
                neighbor_fea = neighbor_fea.squeeze(4)
                # neighbor_fea [num_meshes, in_channel, num_faces//2, 6]

                if pool_idx is not None:
                    # Pool input feature only at pool_idx
                    # Pooling here occurs at the spatial dimension
                    fea = fea[:, :, pool_idx, :]
                    # [num_meshes, in_channel, num_faces//2, 1]

                # Concatenate gathered neighbor features to face_feature, and then find the sum
                copy_fea = fea
                fea = torch.cat([fea, neighbor_fea], 3)
                # fea [num_meshes, in_channel, num_faces//2, 7]

                if self.op_method == 'sum':

                    ''' the sum implementation'''
                    # fea [num_meshes, in_channel, num_faces//2, 7]
                    fea = torch.sum(fea, 3)
                    # fea [num_meshes, in_channel, num_faces//2]
                    conv_fea = self.concat_mlp(fea)
                    # [num_meshes, in_channel, num_faces//2, 7]
                
                elif self.op_method == 'attention':

                    ''' attention implementation '''
                    key = self.key_linear(copy_fea)
                    # key [num_meshes, 128, num_faces//2, 1]
                    query = self.query_linear(fea) # query [num_meshes, 128, num_faces//2, 7]

                    attention = torch.sum(key * query, 1, keepdim=True) # [num_meshes, 1, num_faces//2, 7]
                    attention = attention / math.sqrt(128)
                    attention = self.softmax(attention)

                    # fea [num_meshes, in_channel, num_faces//2, 7]
                    fea = torch.sum(fea * attention, 3)
                    conv_fea = self.concat_mlp(fea)

                else:

                    raise TypeError('not valid op type')

            elif self.gather_range == 'global':
                # fea       [num_meshes,  in_channel, num_faces]
                fea = fea.unsqueeze(3)
                # fea       [num_meshes,  in_channel, num_faces, 1]
                key = self.key_linear(fea)
                # key       [num_meshes,         128, num_faces, 1]
                key = key.squeeze(3)
                # key       [num_meshes,         128, num_faces]
                query = self.query_linear(fea)
                # query     [num_meshes,         128, num_faces, 1]
                query = query.squeeze(3)
                # query     [num_meshes,         128, num_faces]
                query = query.permute(0, 2, 1) 
                # query     [num_meshes,   num_faces,       128]
                attention = torch.bmm(query, key) 
                # attention [num_meshes,   num_faces, num_faces]
                attention = attention / math.sqrt(128)
                # attention [num_meshes,   num_faces, num_faces]

                if pool_idx is not None:
                    # pool_idx [num_pool_faces]
                    self_idx = pool_idx.unsqueeze(0).unsqueeze(2)
                    self_idx = self_idx.to(attention.device)
                    # pool_idx [             1, num_pool_faces, 1]
                    self_idx = self_idx.expand(num_meshes, -1, -1)
                    # self_idx [    num_meshes, num_pool_faces, 1]
                else:
                    self_idx = torch.arange(num_faces, device=attention.device)
                    # self_idx [num_pool_faces]
                    self_idx = self_idx.unsqueeze(0).unsqueeze(2)
                    # self_idx [1, num_faces, 1]
                    self_idx = self_idx.expand(num_meshes, -1, -1)
                    # self_idx [num_meshes, num_faces, 1]

                # ring_n        [num_meshes, num_pool_faces, 6]
                attention_idx = torch.cat([self_idx, ring_n], -1)
                # attention_idx [num_meshes, num_pool_faces, 7]

                mask = torch.zeros_like(attention)
                if pool_idx is not None:
                    mask[
                        torch.arange(num_meshes)[:, None, None],
                        pool_idx[None, :, None],
                        attention_idx,
                    ] = 1
                else:
                    mask[
                        torch.arange(num_meshes)[:, None, None],
                        torch.arange(500)[None, :, None],
                        attention_idx,
                    ] = 1

                if self.gather_method == 'softmax':

                    attention = mask * attention + (1. - mask) * (-1e10)
                    # attention [num_meshes, num_faces, num_faces]
                    attention = self.softmax(attention)

                elif self.gather_method == 'norm':

                    attention = mask * attention
                    attention = nn.functional.normalize(attention, dim=-1)

                # get new feature
                fea = torch.bmm(attention, fea.squeeze(3).permute(0, 2, 1)).permute(0, 2, 1)
                if pool_idx is not None:
                    # Pool input feature only at pool_idx
                    # Pooling here occurs at the spatial dimension
                    fea = fea[:, :, pool_idx]
                conv_fea = self.concat_mlp(fea)

            else:
                raise TypeError('not valid gather range')

        else:
            # fea [num_meshes, in_channel, num_faces]
            fea = fea.unsqueeze(3)
            # fea [num_meshes, in_channel, num_faces, 1]

            # if pool_idx is not None:
            #     # Pool input feature only at pool_idx
            #     # Pooling here occurs at the spatial dimension
            #     fea = fea[:, :, pool_idx, :]
            #     # [num_meshes, in_channel, num_faces//2, 1]

            # # get key and query
            # key = self.key_linear(fea) # key [num_meshes, 128, num_faces//2, 1]
            # key = key.squeeze(3) # key [num_meshes, 128, num_faces//2]
            # query = self.query_linear(fea) # query [num_meshes, 128, num_faces//2, 1]
            # query = query.squeeze(3) # query [num_meshes, 128, num_faces//2]
            # query = query.permute(0, 2, 1) # query [num_meshes, num_faces//2, 128]

            # # attention feature
            # attention = torch.bmm(query, key) # attention [num_meshes, num_face//2, num_faces//2]
            # attention = attention / math.sqrt(128)
            # attention = self.softmax(attention)

            # # get new feature
            # fea = torch.bmm(attention, fea.squeeze(3).permute(0, 2, 1)).permute(0, 2, 1)
            # # fea [num_meshes, in_channel, num_faces//2]
            # conv_fea = self.concat_mlp(fea)




            # fea       [num_meshes,  in_channel, num_faces]
            pos_embed = pos_embed.unsqueeze(3)
            # fea       [num_meshes,  in_channel, num_faces, 1]
            key = self.key_linear(pos_embed)
            # key       [num_meshes,         128, num_faces, 1]
            key = key.squeeze(3)
            # key       [num_meshes,         128, num_faces]
            query = self.query_linear(pos_embed)
            # query     [num_meshes,         128, num_faces, 1]
            query = query.squeeze(3)
            # query     [num_meshes,         128, num_faces]
            query = query.permute(0, 2, 1) 
            # query     [num_meshes,   num_faces,       128]
            attention = torch.bmm(query, key) 
            # attention [num_meshes,   num_faces, num_faces]
            attention = attention / math.sqrt(128)
            # attention [num_meshes,   num_faces, num_faces]

            # if pool_idx is not None:
            #     # pool_idx [num_pool_faces]
            #     self_idx = pool_idx.unsqueeze(0).unsqueeze(2)
            #     self_idx = self_idx.to(attention.device)
            #     # pool_idx [             1, num_pool_faces, 1]
            #     self_idx = self_idx.expand(num_meshes, -1, -1)
            #     # self_idx [    num_meshes, num_pool_faces, 1]
            # else:
            #     self_idx = torch.arange(num_faces, device=attention.device)
            #     # self_idx [num_pool_faces]
            #     self_idx = self_idx.unsqueeze(0).unsqueeze(2)
            #     # self_idx [1, num_faces, 1]
            #     self_idx = self_idx.expand(num_meshes, -1, -1)
            #     # self_idx [num_meshes, num_faces, 1]

            # # ring_n        [num_meshes, num_pool_faces, 6]
            # attention_idx = torch.cat([self_idx, ring_n], -1)
            # # attention_idx [num_meshes, num_pool_faces, 7]

            # mask = torch.zeros_like(attention)
            # if pool_idx is not None:
            #     mask[
            #         torch.arange(num_meshes)[:, None, None],
            #         pool_idx[None, :, None],
            #         attention_idx,
            #     ] = 1
            # else:
            #     mask[
            #         torch.arange(num_meshes)[:, None, None],
            #         torch.arange(500)[None, :, None],
            #         attention_idx,
            #     ] = 1

            if self.gather_method == 'softmax':

                # attention = mask * attention + (1. - mask) * (-1e10)
                # attention [num_meshes, num_faces, num_faces]
                attention = self.softmax(attention)

            elif self.gather_method == 'norm':

                attention = mask * attention
                attention = nn.functional.normalize(attention, dim=-1)

            # get new feature
            fea = torch.bmm(attention, fea.squeeze(3).permute(0, 2, 1)).permute(0, 2, 1)
            if pool_idx is not None:
                # Pool input feature only at pool_idx
                # Pooling here occurs at the spatial dimension
                fea = fea[:, :, pool_idx]
            conv_fea = self.concat_mlp(fea)

        return conv_fea


class ConvFaceBlock(nn.Module):
    """
    Multiple PsuedoConvFaceBlock layers create a PsuedoMeshBlock.
    PsuedoConvFaceBlock is comprised of PsuedoConvFace layers.
    First PsuedoConvFace layer convolves on in_channel to produce "128" channels.
    Second PsuedoConvFace convolves these "128" channels to produce "growth factor" channels.
    These features get concatenated to the original input feature to produce
    "in_channel + growth_factor" channels.
    Note: The original mesh dimensions are maintained for gathering the neighbor features but
    the operations get perfomed only on the pooling indices.
    """
    def __init__(self, in_channel, growth_factor, num_neighbor, **kwargs):
        """
        Args:
        in_channel: number of channels in feature

        growth_factor: number of channels to increase in_channel by

        num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(ConvFaceBlock, self).__init__()
        self.in_channel = in_channel
        self.growth_factor = growth_factor
        self.num_neighbor = num_neighbor
        self.conv_face_1 = ConvFace(in_channel, 128, num_neighbor, **kwargs)
        self.conv_face_2 = ConvFace(128, growth_factor, num_neighbor, **kwargs)

    def forward(self, fea, ring_n, pool_idx, pos_embed):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor]

            pool_idx: indices of faces to be considered for spatial pooling
            [num_faces]//2 OR [num_faces]//4

        Returns:
            conv_block_fea: features produced by ConvFaceBlock layer
            [num_meshes, in_channel + growth_factor, num_faces]
        """
        fea_copy = fea

        device = fea.device
        num_meshes, num_channels, num_faces = fea.size()

        # Convolve
        fea = self.conv_face_1(fea, ring_n, pool_idx, pos_embed)
        if pool_idx is not None:
            # Create placeholder for tensor re-assignment
            fea_placeholder = torch.zeros((num_meshes, fea.shape[1], num_faces), device=device)
            c = torch.arange(fea.shape[1])[None, :, None]
            n = torch.arange(num_meshes)[:, None, None]
            p = pool_idx[None, None, :]
            # Assign values from fea to fea_placeholder at pooling indicies
            # Values at non pooling indices will be zero
            fea_placeholder[n, c, p] = fea
        else:
            fea_placeholder = fea

        # Convolve
        fea = self.conv_face_2(fea_placeholder, ring_n, pool_idx, pos_embed)
        if pool_idx is not None:
            # Create placeholder for tensor re-assignment
            fea_placeholder = torch.zeros((num_meshes, fea.shape[1], num_faces), device=device)
            c = torch.arange(fea.shape[1])[None, :, None]
            n = torch.arange(num_meshes)[:, None, None]
            p = pool_idx[None, None, :]
            # Assign values from fea to fea_placeholder at pooling indicies
            # Values at non pooling indices will be zero
            fea_placeholder[n, c, p] = fea
        else:
            fea_placeholder = fea

        conv_block_fea = torch.cat([fea_copy, fea_placeholder], 1)

        return conv_block_fea


class MeshBlock(nn.ModuleDict):
    """
    Multiple MeshBlock layers create MeshNet2.
    MeshBlock is comprised of several ConvFaceBlock layers.
    """
    def __init__(self, in_channel, num_neighbor, num_block, growth_factor, **kwargs):
        """
        in_channel: number of channels in feature

        growth_factor: number of channels a single ConvFaceBlock increase in_channel by

        num_block: number of ConvFaceBlock layers in a single MeshBlock

        num_neighbor: per faces neighbors in a n-Ring neighborhood.
        """
        super(MeshBlock, self).__init__()

        hidden_channel = in_channel
        for i in range(0, num_block):
            layer = ConvFaceBlock(
                in_channel=hidden_channel, growth_factor=growth_factor,
                num_neighbor=num_neighbor, **kwargs
            )
            hidden_channel = hidden_channel + growth_factor
            self.add_module('meshblock%d' % (i + 1), layer)

    def forward(self, fea, ring_n, pool_idx=None, pos_embed=None):
        """
        Args:
            fea: face features of meshes
            [num_meshes, in_channel, num_faces]

            ring_n: faces in a n-Ring neighborhood
            [num_meshes, num_faces, num_neighbor]

        Returns:
            fea: features produced by MeshBlock layer
            [num_meshes, in_channel + growth_factor * num_block, num_faces]
        """
        if pool_idx is not None:
            ring_n = ring_n[:, pool_idx, :]

        for _, layer in self.items():
            fea = layer(fea, ring_n, pool_idx, pos_embed)
        return fea


class NetworkBeforeLinear(nn.Module):
    def __init__(self, num_faces, cfg, num_cls, pool_rate, **kwargs):
        super(Network, self).__init__()
        self.pool_rate = pool_rate

        # feature engineering
        self.point_descriptor = PointDescriptor(
            num_kernel=cfg['num_kernel']
        )
        self.normal_descriptor = NormalDescriptor(
            num_kernel=cfg['num_kernel']
        )
        self.conv_surface_1 = ConvSurface(
            num_faces=num_faces, num_neighbor=3, cfg=cfg['ConvSurface']
        )
        self.conv_surface_2 = ConvSurface(
            num_faces=num_faces, num_neighbor=6, cfg=cfg['ConvSurface']
        )
        self.conv_surface_3 = ConvSurface(
            num_faces=num_faces, num_neighbor=12, cfg=cfg['ConvSurface']
        )

        ''' Option 1: transformer
        '''
        # self.te = GeneralTransformerNoPE(
        #     msl=600, hd=320, nh=4, nl=4
        # )

        ''' Option 2: face conv
        '''
        blocks = cfg['MeshBlock']['blocks']
        in_channel = cfg['num_kernel'] * 2 + cfg['ConvSurface']['num_kernel'] * 3

        self.mesh_block_1 = MeshBlock(
            in_channel=in_channel, num_block=blocks[0],
            growth_factor=cfg['num_kernel'], num_neighbor=3, **kwargs
        )
        in_channel = in_channel + blocks[0] * cfg['num_kernel']
        self.max_pool_fea_1 = MaxPoolFaceFeature(in_channel=in_channel, num_neighbor=3)

        self.mesh_block_2 = MeshBlock(
            in_channel=in_channel, num_block=blocks[1],
            growth_factor=cfg['num_kernel'], num_neighbor=6
        )
        in_channel = in_channel + blocks[1] * cfg['num_kernel']
        self.max_pool_fea_2 = MaxPoolFaceFeature(in_channel=in_channel, num_neighbor=6)

        self.mesh_block_3 = MeshBlock(
            in_channel=in_channel, num_block=blocks[2],
            growth_factor=cfg['num_kernel'], num_neighbor=12
        )
        in_channel = in_channel + blocks[2] * cfg['num_kernel']

        # TODO need to fit to feature do it later
        # in_channel = 320
        # self.classifier = nn.Sequential(
        #     nn.Linear(in_channel, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(256, num_cls)
        # )
        embed_kwargs = {
            'include_input' : True,
            'input_dims' : 3,
            'max_freq_log2' : 5 - 1,
            'num_freqs' : 5,
            'periodic_fns' : [torch.sin, torch.cos],
        }
        self.pos_embeder = Embedder(**embed_kwargs)
        self.nor_embeder = Embedder(**embed_kwargs)

    def forward(
        self, verts, faces, centers, normals, ring_1, ring_2, ring_3,
        **kwargs
    ):
        """
        Args:
            verts: padded mesh vertices
            [num_meshes, ?, 3]

            faces: faces in mesh/es
            [num_meshes, num_faces, 3]

            centers: face center of mesh/es
            [num_meshes, num_faces, 3]

            normals: face normals of mesh/es
            [num_meshes, num_faces, 3]

            ring_1: 1st Ring neighbors of faces
            [num_meshes, num_faces, 3]

            ring_2: 2nd Ring neighbors of faces
            [num_meshes, num_faces, 6]

            ring_3: 3rd Ring neighbors of faces
            [num_meshes, num_faces, 12]

        Returns:
            cls: predicted class of the input mesh/es
        """
        # Face center features
        points_fea = self.point_descriptor(centers=centers)

        # Face normal features
        normals_fea = self.normal_descriptor(normals=normals)

        # Surface features from 1-Ring neighborhood around a face
        surface_fea_1 = self.conv_surface_1(verts=verts,
                                            faces=faces,
                                            ring_n=ring_1,
                                            centers=centers)

        # Surface features from 2-Ring neighborhood around a face
        surface_fea_2 = self.conv_surface_2(verts=verts,
                                            faces=faces,
                                            ring_n=ring_2,
                                            centers=centers)

        # Surface features from 3-Ring neighborhood around a face
        surface_fea_3 = self.conv_surface_3(verts=verts,
                                            faces=faces,
                                            ring_n=ring_3,
                                            centers=centers)

        # Concatenate spatial and structural features
        fea_in = torch.cat([points_fea, surface_fea_1, surface_fea_2, surface_fea_3, normals_fea], 1)
        # fea_in: [batch_size, 320, 500]

        pos_embed = self.pos_embeder.embed(centers.permute(0, 2, 1)).permute(0, 2, 1)
        nor_embed = self.nor_embeder.embed(normals.permute(0, 2, 1)).permute(0, 2, 1)
        pos_embed = torch.cat((pos_embed, nor_embed), -2)

        ''' option 1: transformer '''
        # fea_in = fea_in.permute(0, 2, 1)
        # y = self.te(fea_in)
        # fea, _ = torch.max(y, 1)

        ''' option 2: face conv '''
        # Mesh block 1 features
        fea = self.mesh_block_1(fea=fea_in, ring_n=ring_1, pos_embed=pos_embed)

        # Max pool features
        fea = self.max_pool_fea_1(fea=fea, ring_n=ring_1)

        # Randomly select pooling indicies. Face indices not in pooling_idx will not be considered by
        # further layers.
        # Note: pooling_idx is same for all meshes and size of the orginal tensor does not change
        pool_idx = torch.randperm(ring_2.shape[1])[:ring_2.shape[1]//self.pool_rate]

        # Sort the index for correct tensor re-assignment in PsuedoMeshBlock
        pool_idx, _ = torch.sort(pool_idx)

        # Mesh block 2 features
        fea = self.mesh_block_2(fea=fea, ring_n=ring_2, pool_idx=pool_idx, pos_embed=pos_embed)
        # fea = self.mesh_block_2(fea=fea, ring_n=ring_2)

        # Max pool features
        fea = self.max_pool_fea_2(fea=fea, ring_n=ring_2)

        # Randomly subset pooling indicies from initial pool_idx
        pool_idx_idx = torch.randperm(pool_idx.shape[0])[:pool_idx.shape[0]//self.pool_rate]
        pool_idx = pool_idx[pool_idx_idx]
        pool_idx, _ = torch.sort(pool_idx)

        # Mesh block 3 features
        fea = self.mesh_block_3(fea=fea, ring_n=ring_3, pool_idx=pool_idx, pos_embed=pos_embed)
        # fea = self.mesh_block_3(fea=fea, ring_n=ring_3)

        # Only consider the pool_idx, global features
        fea = fea[:, :, pool_idx]

        fea = torch.max(fea, dim=2)[0]
        fea = fea.reshape(fea.size(0), -1)

        ''' classifier '''
        # logit = self.classifier(fea)

        return fea


class Network(nn.Module):
    def __init__(self, num_faces, cfg, num_cls, pool_rate, **kwargs):
        super(Network, self).__init__()
        self.pool_rate = pool_rate

        # feature engineering
        self.point_descriptor = PointDescriptor(
            num_kernel=cfg['num_kernel']
        )
        self.normal_descriptor = NormalDescriptor(
            num_kernel=cfg['num_kernel']
        )
        self.conv_surface_1 = ConvSurface(
            num_faces=num_faces, num_neighbor=3, cfg=cfg['ConvSurface']
        )
        self.conv_surface_2 = ConvSurface(
            num_faces=num_faces, num_neighbor=6, cfg=cfg['ConvSurface']
        )
        self.conv_surface_3 = ConvSurface(
            num_faces=num_faces, num_neighbor=12, cfg=cfg['ConvSurface']
        )

        ''' Option 1: transformer
        '''
        # self.te = GeneralTransformerNoPE(
        #     msl=600, hd=320, nh=4, nl=4
        # )

        ''' Option 2: face conv
        '''
        blocks = cfg['MeshBlock']['blocks']
        in_channel = cfg['num_kernel'] * 2 + cfg['ConvSurface']['num_kernel'] * 3

        self.mesh_block_1 = MeshBlock(
            in_channel=in_channel, num_block=blocks[0],
            growth_factor=cfg['num_kernel'], num_neighbor=3, **kwargs
        )
        in_channel = in_channel + blocks[0] * cfg['num_kernel']
        self.max_pool_fea_1 = MaxPoolFaceFeature(in_channel=in_channel, num_neighbor=3)

        self.mesh_block_2 = MeshBlock(
            in_channel=in_channel, num_block=blocks[1],
            growth_factor=cfg['num_kernel'], num_neighbor=6
        )
        in_channel = in_channel + blocks[1] * cfg['num_kernel']
        self.max_pool_fea_2 = MaxPoolFaceFeature(in_channel=in_channel, num_neighbor=6)

        self.mesh_block_3 = MeshBlock(
            in_channel=in_channel, num_block=blocks[2],
            growth_factor=cfg['num_kernel'], num_neighbor=12
        )
        in_channel = in_channel + blocks[2] * cfg['num_kernel']

        # TODO need to fit to feature do it later
        # in_channel = 320
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
        embed_kwargs = {
            'include_input' : True,
            'input_dims' : 3,
            'max_freq_log2' : 5 - 1,
            'num_freqs' : 5,
            'periodic_fns' : [torch.sin, torch.cos],
        }
        self.pos_embeder = Embedder(**embed_kwargs)
        self.nor_embeder = Embedder(**embed_kwargs)

    def forward(
        self, verts, faces, centers, normals, ring_1, ring_2, ring_3,
        **kwargs
    ):
        """
        Args:
            verts: padded mesh vertices
            [num_meshes, ?, 3]

            faces: faces in mesh/es
            [num_meshes, num_faces, 3]

            centers: face center of mesh/es
            [num_meshes, num_faces, 3]

            normals: face normals of mesh/es
            [num_meshes, num_faces, 3]

            ring_1: 1st Ring neighbors of faces
            [num_meshes, num_faces, 3]

            ring_2: 2nd Ring neighbors of faces
            [num_meshes, num_faces, 6]

            ring_3: 3rd Ring neighbors of faces
            [num_meshes, num_faces, 12]

        Returns:
            cls: predicted class of the input mesh/es
        """
        # Face center features
        points_fea = self.point_descriptor(centers=centers)

        # Face normal features
        normals_fea = self.normal_descriptor(normals=normals)

        # Surface features from 1-Ring neighborhood around a face
        surface_fea_1 = self.conv_surface_1(verts=verts,
                                            faces=faces,
                                            ring_n=ring_1,
                                            centers=centers)

        # Surface features from 2-Ring neighborhood around a face
        surface_fea_2 = self.conv_surface_2(verts=verts,
                                            faces=faces,
                                            ring_n=ring_2,
                                            centers=centers)

        # Surface features from 3-Ring neighborhood around a face
        surface_fea_3 = self.conv_surface_3(verts=verts,
                                            faces=faces,
                                            ring_n=ring_3,
                                            centers=centers)

        # Concatenate spatial and structural features
        fea_in = torch.cat([points_fea, surface_fea_1, surface_fea_2, surface_fea_3, normals_fea], 1)
        # fea_in: [batch_size, 320, 500]

        pos_embed = self.pos_embeder.embed(centers.permute(0, 2, 1)).permute(0, 2, 1)
        nor_embed = self.nor_embeder.embed(normals.permute(0, 2, 1)).permute(0, 2, 1)
        pos_embed = torch.cat((pos_embed, nor_embed), -2)

        ''' option 1: transformer '''
        # fea_in = fea_in.permute(0, 2, 1)
        # y = self.te(fea_in)
        # fea, _ = torch.max(y, 1)

        ''' option 2: face conv '''
        # Mesh block 1 features
        fea = self.mesh_block_1(fea=fea_in, ring_n=ring_1, pos_embed=pos_embed)

        # Max pool features
        fea = self.max_pool_fea_1(fea=fea, ring_n=ring_1)

        # Randomly select pooling indicies. Face indices not in pooling_idx will not be considered by
        # further layers.
        # Note: pooling_idx is same for all meshes and size of the orginal tensor does not change
        pool_idx = torch.randperm(ring_2.shape[1])[:ring_2.shape[1]//self.pool_rate]

        # Sort the index for correct tensor re-assignment in PsuedoMeshBlock
        pool_idx, _ = torch.sort(pool_idx)

        # Mesh block 2 features
        fea = self.mesh_block_2(fea=fea, ring_n=ring_2, pool_idx=pool_idx, pos_embed=pos_embed)
        # fea = self.mesh_block_2(fea=fea, ring_n=ring_2)

        # Max pool features
        fea = self.max_pool_fea_2(fea=fea, ring_n=ring_2)

        # Randomly subset pooling indicies from initial pool_idx
        pool_idx_idx = torch.randperm(pool_idx.shape[0])[:pool_idx.shape[0]//self.pool_rate]
        pool_idx = pool_idx[pool_idx_idx]
        pool_idx, _ = torch.sort(pool_idx)

        # Mesh block 3 features
        fea = self.mesh_block_3(fea=fea, ring_n=ring_3, pool_idx=pool_idx, pos_embed=pos_embed)
        # fea = self.mesh_block_3(fea=fea, ring_n=ring_3)

        # Only consider the pool_idx, global features
        fea = fea[:, :, pool_idx]

        fea = torch.max(fea, dim=2)[0]
        fea = fea.reshape(fea.size(0), -1)

        ''' classifier '''
        logit = self.classifier(fea)

        return logit
