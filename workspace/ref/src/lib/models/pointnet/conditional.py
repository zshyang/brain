import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model.pointnet.util import PointNetFeat


class CNet(nn.Module):
    def __init__(
        self, 
        have_da_embed: bool,
        da_ratio: float,
        have_fc: bool,
        num_channel=3,
        **kwargs
    ):
        super(
            CNet, self
        ).__init__()

        self.feat = PointNetFeat(
            global_feat=False,
            feature_transform=False,
            channel=num_channel,
            **kwargs
        )

        self.have_da_embed = \
        have_da_embed
        self.da_ratio = da_ratio
        if have_da_embed:
            self.da_embed = nn.Embedding(
                num_embeddings=2,
                embedding_dim=1024
            )

        self.have_fc = have_fc
        if have_fc:
            self.linear0 = nn.Linear(
                1024, 256
            )
            self.linear1 = nn.Linear(
                256, 256
            )

    def forward(self, x):
        # x = x.transpose(1, 2)
        bs, _, num_points = x[0].size()

        _, tm, tm1, y = self.feat(x[0])
        # tm1 : [b, 64, 64]

        # y = torch.cat(
        #     (y, self.da_embed(x[1])), 1
        # )
        if self.have_da_embed:
            y = y + self.da_ratio * \
            self.da_embed(
                x[1]
            )
        
        if self.have_fc:
            y = self.linear0(y)
            y = F.relu(y)
            y = self.linear1(y)

        return y


class DENet(nn.Module):
    def __init__(
        self, num_channel=3,
        **kwargs
    ):
        super(
            DENet, self
        ).__init__()

        self.feat = PointNetFeat(
            global_feat=False,
            feature_transform=False,
            channel=num_channel,
            **kwargs
        )

        self.linear0 = nn.Linear(
            256, 256
        )
        self.linear1 = nn.Linear(
            256, 2
        )

    def forward(self, x):
        # x = x.transpose(1, 2)
        bs, _, num_points = x.size()

        _, tm, tm1, y = self.feat(x)

        class_feature = y[:, :768]
        da_feature = y[:, 768:]

        y = self.linear0(da_feature)
        y = F.relu(y)
        y = self.linear1(y)

        return class_feature, y


class FCNet(CNet):
    def __init__(
        self, 
        have_da_embed: bool,
        da_ratio: float,
        have_fc: bool,
        num_channel=3,
        **kwargs
    ):
        super(
            FCNet, self
        ).__init__(
            have_da_embed=have_da_embed,
            da_ratio=da_ratio,
            have_fc=have_fc
        )

    def forward(self, x, qc):
        # x = x.transpose(1, 2)
        bs, _, num_points = x.size()

        _, tm, tm1, y = self.feat(x)
        # tm1 : [b, 64, 64]

        # y = torch.cat(
        #     (y, self.da_embed(x[1])), 1
        # )
        if self.have_da_embed:
            y = y + self.da_ratio * \
            self.da_embed(
                qc
            )
        
        if self.have_fc:
            y = self.linear0(y)
            y = F.relu(y)
            y = self.linear1(y)

        return y
