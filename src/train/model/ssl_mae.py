''' model for modelnet40 with meshnet2 as encoder
self-supervised learning and with masked autoencoder.

class:
    c00 EmptyEmbedding(nn.Module)
        00 __init__(self, embed_dim)
        01 forward(self)
    c01 MaskedMeshNet(nn.Module)
        00 __init__(self, num_faces, cfg, num_cls, pool_rate, mask_percentage=0.25, **kwargs)
        01 forward(self, verts, faces, centers, normals, ring_1, ring_2, ring_3, **kwargs)
    c02 Network(nn.Module)
        00 __init__(self, en_config={}, de_config={}, **kwargs)
        01 forward(self, verts, faces, centers, normals, ring_1, ring_2, ring_3, **kwargs)

author:
    zhangsihao yang

date:
    20220911

logs:
    20220911
        file    created
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
from torch.nn.parameter import Parameter


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


class MaskedMeshNetPerFaceFea(nn.Module):
    def __init__(self, num_faces, cfg, num_cls, pool_rate, mask_percentage=0.25, **kwargs):
        super(MaskedMeshNetPerFaceFea, self).__init__()
        self.pool_rate = pool_rate
        self.mask_percentage = mask_percentage

        # empty token
        self.empty_embedding = EmptyEmbedding(320)

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

    def forward(self, verts, faces, centers, normals, ring_1, ring_2, ring_3, **kwargs):
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

        # mask the input feature
        batch_size, _, num_faces = fea_in.size()
        mask_num_faces = int(self.mask_percentage * num_faces)
        mask_idx = torch.randperm(num_faces)[:mask_num_faces]
        fea_in[:, :, mask_idx] = torch.unsqueeze(self.empty_embedding(), -1)

        # Mesh block 1 features
        fea = self.mesh_block_1(fea=fea_in, ring_n=ring_1)

        # Max pool features
        fea = self.max_pool_fea_1(fea=fea, ring_n=ring_1)

        # Randomly select pooling indicies. Face indices not in pooling_idx will not be considered by
        # further layers.
        # Note: pooling_idx is same for all meshes and size of the orginal tensor does not change
        pool_idx = torch.randperm(ring_2.shape[1])[:ring_2.shape[1]//self.pool_rate]

        # Sort the index for correct tensor re-assignment in PsuedoMeshBlock
        pool_idx, _ = torch.sort(pool_idx)

        # Mesh block 2 features
        fea = self.mesh_block_2(fea=fea, ring_n=ring_2, pool_idx=pool_idx)
        # fea = self.mesh_block_2(fea=fea, ring_n=ring_2)

        # Max pool features
        fea = self.max_pool_fea_2(fea=fea, ring_n=ring_2)

        # Randomly subset pooling indicies from initial pool_idx
        pool_idx_idx = torch.randperm(pool_idx.shape[0])[:pool_idx.shape[0]//self.pool_rate]
        pool_idx = pool_idx[pool_idx_idx]
        pool_idx, _ = torch.sort(pool_idx)

        # Mesh block 3 features
        fea = self.mesh_block_3(fea=fea, ring_n=ring_3, pool_idx=pool_idx)
        # fea = self.mesh_block_3(fea=fea, ring_n=ring_3)

        # Only consider the pool_idx, global features
        fea = fea[:, :, pool_idx]

        globel_fea = torch.max(fea, dim=2)[0]
        globel_fea = globel_fea.view(fea.size(0), 1024, 1)
        globel_fea = globel_fea.expand(fea.size(0), 1024, 1024)
        fea = torch.cat((fea, globel_fea), 1) # [bs, 2048, 1024]
        # fea = fea.reshape(fea.size(0), -1)

        return fea


class MaskedMeshNet(nn.Module):
    def __init__(self, num_faces, cfg, num_cls, pool_rate, mask_percentage=0.25, **kwargs):
        super(MaskedMeshNet, self).__init__()
        self.pool_rate = pool_rate
        self.mask_percentage = mask_percentage

        # empty token
        self.empty_embedding = EmptyEmbedding(320)

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

    def forward(self, verts, faces, centers, normals, ring_1, ring_2, ring_3, **kwargs):
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

        # mask the input feature
        batch_size, _, num_faces = fea_in.size()
        mask_num_faces = int(self.mask_percentage * num_faces)
        mask_idx = torch.randperm(num_faces)[:mask_num_faces]
        fea_in[:, :, mask_idx] = torch.unsqueeze(self.empty_embedding(), -1)

        # Mesh block 1 features
        fea = self.mesh_block_1(fea=fea_in, ring_n=ring_1)

        # Max pool features
        fea = self.max_pool_fea_1(fea=fea, ring_n=ring_1)

        # Randomly select pooling indicies. Face indices not in pooling_idx will not be considered by
        # further layers.
        # Note: pooling_idx is same for all meshes and size of the orginal tensor does not change
        pool_idx = torch.randperm(ring_2.shape[1])[:ring_2.shape[1]//self.pool_rate]

        # Sort the index for correct tensor re-assignment in PsuedoMeshBlock
        pool_idx, _ = torch.sort(pool_idx)

        # Mesh block 2 features
        fea = self.mesh_block_2(fea=fea, ring_n=ring_2, pool_idx=pool_idx)
        # fea = self.mesh_block_2(fea=fea, ring_n=ring_2)

        # Max pool features
        fea = self.max_pool_fea_2(fea=fea, ring_n=ring_2)

        # Randomly subset pooling indicies from initial pool_idx
        pool_idx_idx = torch.randperm(pool_idx.shape[0])[:pool_idx.shape[0]//self.pool_rate]
        pool_idx = pool_idx[pool_idx_idx]
        pool_idx, _ = torch.sort(pool_idx)

        # Mesh block 3 features
        fea = self.mesh_block_3(fea=fea, ring_n=ring_3, pool_idx=pool_idx)
        # fea = self.mesh_block_3(fea=fea, ring_n=ring_3)

        # Only consider the pool_idx, global features
        fea = fea[:, :, pool_idx]

        fea = torch.max(fea, dim=2)[0]
        fea = fea.reshape(fea.size(0), -1)

        return fea


class Network(nn.Module):
    def __init__(self, en_config={}, de_config={}, **kwargs):
        super(Network, self).__init__()

        self.encoder = MaskedMeshNet(**en_config)
        self.decoder = PointNetDecoder(**de_config)

    def forward(
        self, verts, faces, centers, normals, ring_1, ring_2, ring_3, **kwargs
    ):
        fea = self.encoder(verts, faces, centers, normals, ring_1, ring_2, ring_3)
        y = self.decoder(fea)
        return y, fea
