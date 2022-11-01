''' sparse convolution unet

name convention:
function:
    safe_half(integer)

class:
author:
    Zhangsihao Yang

date:
    20220607

logs:
    20220607
        add function safe_half(integer)
'''
import torch.nn as nn
import spconv
import torch.nn.functional as F
import torch
# from lib.config import cfg


class RefNetwork(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.c = nn.Embedding(6890, 16)
        self.xyzc_net = SparseConvNet()

        self.latent = nn.Embedding(256, 128)

        self.actvn = nn.ReLU()

        self.fc_0 = nn.Conv1d(352, 256, 1)
        self.fc_1 = nn.Conv1d(256, 256, 1)
        self.fc_2 = nn.Conv1d(256, 256, 1)
        self.alpha_fc = nn.Conv1d(256, 1, 1)

        self.feature_fc = nn.Conv1d(256, 256, 1)
        self.latent_fc = nn.Conv1d(384, 256, 1)
        self.view_fc = nn.Conv1d(346, 128, 1)
        self.rgb_fc = nn.Conv1d(128, 3, 1)

    def forward(self, sp_input, tgrid_coords, pgrid_coords, viewdir,
                light_pts):
        coord = sp_input['coord']
        out_sh = sp_input['out_sh']
        batch_size = sp_input['batch_size']

        pgrid_coords = pgrid_coords[:, None, None]

        code = self.c(torch.arange(0, 6890).to(tgrid_coords.device))
        xyzc = spconv.SparseConvTensor(code, coord, out_sh, batch_size)

        xyzc_features = self.xyzc_net(xyzc, tgrid_coords, pgrid_coords)

        net = self.actvn(self.fc_0(xyzc_features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))

        alpha = self.alpha_fc(net)

        features = self.feature_fc(net)

        latent = self.latent(sp_input['i'])
        latent = latent[..., None].expand(*latent.shape, net.size(2))
        features = torch.cat((features, latent), dim=1)
        features = self.latent_fc(features)

        viewdir = viewdir.transpose(1, 2)
        light_pts = light_pts.transpose(1, 2)
        features = torch.cat((features, viewdir, light_pts), dim=1)
        net = self.actvn(self.view_fc(features))
        rgb = self.rgb_fc(net)

        raw = torch.cat((rgb, alpha), dim=1)
        raw = raw.transpose(1, 2)

        return raw


def safe_half(integer):
    half = integer // 2
    if (half + half) != integer:
        raise 'the division by half is not safe'
    return half


def merge_two_sparse_tensor(tensor0, tensor1):
    assert tensor0.dense().shape == tensor1.dense().shape
    indices = tensor0.indices.long()
    dense1 = tensor1.dense()
    feature1 = dense1[indices[:, 0], :, indices[:, 1], indices[:, 2], indices[:, 3]]
    tensor0.features = tensor0.features + feature1
    return tensor0


class Network(nn.Module):
    '''
    '''
    def __init__(self, in_channels, out_channels, **kwargs):
        '''
        name convention:
            d = down
            u = up
            0001 = 1
            0010 = 2
            1010 = -2
            

        '''
        super(Network, self).__init__()

        # encoder
        self.conv0001 = single_conv(in_channels, out_channels, 'conv0001')

        self.conv0010 = double_conv(out_channels, out_channels, 'conv0010')
        self.conv0010d = stride_conv(out_channels, out_channels, 'conv0010d')
        # self.conv0010u = up_conv(out_channels, out_channels, 'conv0010u')

        self.conv0011 = double_conv(out_channels, out_channels * 2, 'conv0011')
        self.conv0011d = stride_conv(out_channels * 2, out_channels * 2, 'conv0011d')
        # self.conv0011u = up_conv(out_channels * 2, out_channels, 'conv0011u')

        self.conv0100 = double_conv(out_channels * 2, out_channels * 4, 'conv0100')
        self.conv0100d = stride_conv(out_channels * 4, out_channels * 4, 'conv0100d')
        # self.conv0100u = up_conv(out_channels * 4, out_channels * 2, 'conv0100u')

        self.conv0101 = double_conv(out_channels * 4, out_channels * 4, 'conv0101')
        self.conv0101d = stride_conv(out_channels * 4, out_channels * 4, 'conv0101d')

        # decoder
        self.conv1101 = double_conv(out_channels * 4, out_channels * 4, 'conv1101')
        self.conv1101u = up_conv(out_channels * 4, out_channels * 4, 'conv1101u')

        self.conv1100 = double_conv(out_channels * 4, out_channels * 2, 'conv1100')
        self.conv1100u = up_conv(out_channels * 2, out_channels * 2, 'conv1100u')

        self.conv1011 = double_conv(out_channels * 2, out_channels, 'conv1011')
        self.conv1011u = up_conv(out_channels, out_channels, 'conv1011u')

        self.conv1010 = double_conv(out_channels, out_channels, 'conv1010')
        self.conv1010u = up_conv(out_channels, out_channels, 'conv1010u')

        self.conv1001 = single_conv(out_channels, out_channels, 'conv1001')

    def forward(self, x):
        # encoder
        # (batch_size, in_channels, n, n, n) ==> (batch_size, out_channels, n, n, n)
        out = self.conv0001(x)

        # (batch_size, out_channels, n, n, n) ==> (batch_size, out_channels, n, n, n)
        out = self.conv0010(out)
        # (batch_size, out_channels, n, n, n) ==> (batch_size, out_channels, n / 2, n / 2, n /2)
        out = self.conv0010d(out)
        out2 = out

        # (batch_size, out_channels, n / 2, n / 2, n /2) ==> (batch_size, out_channels * 2, n / 2, n / 2, n / 2)
        out = self.conv0011(out) 
        # (batch_size, out_channels*2, n/2, n/2, n/2) ==> (batch_size, out_channels*2, n/4, n/4, n/4)
        out = self.conv0011d(out)
        out3 = out

        # (batch_size, out_channels*2, n/4, n/4, n/4) ==> (batch_size, out_channels*4, n/4, n/4, n/4)
        out = self.conv0100(out)
        # (batch_size, out_channels*2, n/4, n/4, n/4) ==> (batch_size, out_channels*4, n/8, n/8, n/8)
        out = self.conv0100d(out)
        out4 = out

        # (batch_size, out_channels*4, n/8, n/8, n/8) ==> (batch_size, out_channels*4, n/8, n/8, n/8)
        out = self.conv0101(out)
        # (batch_size, out_channels*4, n/8, n/8, n/8) ==> (batch_size, out_channels*4, n/g, n/g, n/g)
        out = self.conv0101d(out)

        # decoder
        # (batch_size, out_channels*4, n/g, n/g, n/g) ==> (batch_size, out_channels*4, n/g, n/g, n/g)
        out = self.conv1101(out)
        # (batch_size, out_channels*4, n/g, n/g, n/g) ==> (batch_size, out_channels*4, n/8, n/8, n/8)
        out = self.conv1101u(out)

        out = merge_two_sparse_tensor(out4, out)
        # (batch_size, out_channels*4, n/8, n/8, n/8) ==> (batch_size, out_channels*2, n/8, n/8, n/8)
        out = self.conv1100(out)
        # (batch_size, out_channels*2, n/8, n/8, n/8) ==> (batch_size, out_channels*2, n/4, n/4, n/4)
        out = self.conv1100u(out)
        

        out = merge_two_sparse_tensor(out3, out)
        # (batch_size, out_channels*2, n/4, n/4, n/4) ==> (batch_size, out_channels, n/4, n/4, n/4)
        out = self.conv1011(out)
        # (batch_size, out_channels, n/4, n/4, n/4) ==> (batch_size, out_channels, n/2, n/2, n/2)
        out = self.conv1011u(out)

        out = merge_two_sparse_tensor(out2, out)
        # (batch_size, out_channels, n/2, n/2, n/2) ==> (batch_size, out_channels, n/2, n/2, n/2)
        out = self.conv1010(out)
        # (batch_size, out_channels, n/2, n/2, n/2) ==> (batch_size, out_channels, n, n, n)
        out = self.conv1010u(out)

        # (batch_size, out_channels, n, n, n) ==> (batch_size, out_channels, n, n, n)
        out = self.conv1001(out)

        return out


class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(16, 16, 'subm0')
        self.down0 = stride_conv(16, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x, tgrid_coords, pgrid_coords):
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        feature_1 = F.grid_sample(net1,
                                  tgrid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        feature_1 = F.grid_sample(feature_1,
                                  pgrid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        feature_2 = F.grid_sample(net2,
                                  tgrid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        feature_2 = F.grid_sample(feature_2,
                                  pgrid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        feature_3 = F.grid_sample(net3,
                                  tgrid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        feature_3 = F.grid_sample(feature_3,
                                  pgrid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()
        feature_4 = F.grid_sample(net4,
                                  tgrid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)
        feature_4 = F.grid_sample(feature_4,
                                  pgrid_coords,
                                  padding_mode='zeros',
                                  align_corners=True)

        features = torch.cat((feature_1, feature_2, feature_3, feature_4),
                             dim=1)
        features = features.view(features.size(0), -1, features.size(4))

        return features


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        # nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        # nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            2,
                            2,
                            # padding=1,
                            bias=False,
                            indice_key=indice_key),
        # nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), 
        nn.ReLU()
    )


def up_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConvTranspose3d(in_channels,
                                     out_channels,
                                     2,
                                     2,
                                    #  padding=1,
                                     bias=False,
                                     indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU()
    )
    