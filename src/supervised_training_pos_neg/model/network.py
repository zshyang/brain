''' 
author:
    zhangsihao yang

logs:
    20230222: file created
'''
import torch
import torch.nn as nn
from model.block import ConvSurface, NormalDescriptor, PointDescriptor


def safe_get(dict_object, key):
    return_object = dict_object.get(key, None)
    if return_object is None:
        raise ValueError(f'{key} is None')
    return return_object


class Network(nn.Module):
    ''' a very simple network just to combine three part together
    PointDescriptor, 
    NormalDescriptor, 
    ConvSurfaceBlock
    '''

    def __init__(self, en_config, **kwargs):
        super(Network, self).__init__()

        num_faces = safe_get(en_config, 'num_faces')

        # ---- feature extractor
        cfg = safe_get(en_config, 'cfg')
        num_kernel = safe_get(cfg, 'num_kernel')
        conv_surface_cfg = safe_get(cfg, 'ConvSurface')
        conv_surface_num_kernel = safe_get(conv_surface_cfg, 'num_kernel')

        self.point_descriptor = PointDescriptor(num_kernel)
        self.normal_descriptor = NormalDescriptor(num_kernel)
        self.conv_surface_1 = ConvSurface(num_faces, 3, conv_surface_cfg)

        in_channel = num_kernel * 2 + conv_surface_num_kernel

        # ---- classifier
        num_cls = safe_get(en_config, 'num_cls')

        self.classifier = nn.Sequential(
            nn.Linear(in_channel, num_cls)
        )

    def forward(self, verts, faces, centers, normals, ring_1, ring_2, ring_3, **kwargs):
        points_fea = self.point_descriptor(centers=centers)

        normals_fea = self.normal_descriptor(normals=normals)

        surface_fea_1 = self.conv_surface_1(verts, faces, ring_1, centers)

        fea_in = torch.cat([points_fea, surface_fea_1, normals_fea], 1)

        # ---- find max
        fea = torch.max(fea_in, dim=2)[0]

        y = self.classifier(fea)

        return y


if __name__ == '__main__':
    pass
