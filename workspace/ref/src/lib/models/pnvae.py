'''
Author: Zhangsihao Yang
Date: 2022-03-17
References:
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
'''
import torch
import torch.nn as nn

from lib.model.pointnet import PointNetfeat


class PointNetDecoder(nn.Module):
    def __init__(self, bneck_size):
        super(PointNetDecoder, self).__init__()
        self.linear0 = nn.Linear(bneck_size, 1024)
        self.linear1 = nn.Linear(1024, 2048)
        self.linear2 = nn.Linear(2048, 6144)
        self.elu = nn.ELU()

    def forward(self, cls_embeddings):
        batch_size = cls_embeddings.shape[0]
        x = self.linear0(cls_embeddings)
        x = self.elu(x)
        x = self.linear1(x)
        x = self.elu(x)
        x = self.linear2(x)
        point_cloud = x.reshape(batch_size, -1, 3)
        return point_cloud


class PointNetVAE(nn.Module):
    def __init__(self, is_vae, bneck_size):
        super(PointNetVAE, self).__init__()
        self.is_vae = is_vae

        self.feat = PointNetfeat(
            global_feat=True, feature_transform=False,
            bneck_size=bneck_size)
        self.decoder = PointNetDecoder(bneck_size)

        if self.is_vae:
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.fc_mean = nn.Linear(bneck_size, bneck_size)
            self.fc_var = nn.Linear(bneck_size, bneck_size)

    def _reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        '''
        Args:
            x is the input point cloud with shape (B, N, 3)

        Returns:
            y is (B, N, 3)
            mean (B, b_s)
            log_var (B, b_s)
        '''

        x = x.transpose(1, 2)
        x, _, _ = self.feat(x)

        mean = None
        log_var = None
        if self.is_vae:
            mean = self.fc_mean(self.LeakyReLU(x))
            log_var  = self.fc_var(self.LeakyReLU(x))
            x = self._reparameterization(
                mean, torch.exp(0.5 * log_var))
        else:
            mean = x

        y = self.decoder(x)

        return y, mean, log_var
