import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def feature_transform_regularizer(
    trans
):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(
        torch.norm(
            torch.bmm(
                trans,
                trans.transpose(2,1)
            ) - I, dim=(1,2)
        )
    )
    return loss


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()

        self.conv1 = torch.nn.Conv1d(
            k, 64, 1
        )
        self.conv2 = torch.nn.Conv1d(
            64, 128, 1
        )
        self.conv3 = torch.nn.Conv1d(
            128, 1024, 1
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(
            256, k * k
        )

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]

        x = F.relu(
            self.bn1(self.conv1(x))
        )
        x = F.relu(
            self.bn2(self.conv2(x))
        )
        x = F.relu(
            self.bn3(self.conv3(x))
        )

        x = torch.max(
            x, 2, keepdim=True
        )[0]
        x = x.view(-1, 1024)

        x = F.relu(
            self.bn4(self.fc1(x))
        )
        x = F.relu(
            self.bn5(self.fc2(x))
        )
        x = self.fc3(x)

        iden = Variable(
            torch.from_numpy(
                np.eye(
                    self.k
                ).flatten().astype(
                    np.float32
                )
            )
        ).view(
            1, self.k * self.k
        ).repeat(batchsize,1)

        if x.is_cuda:
            iden = iden.cuda()

        x = x + iden
        x = x.view(-1, self.k, self.k)

        return x


class PointNetFeat(nn.Module):
    def __init__(
        self, global_feat=True,
        feature_transform=False,
        bneck_size=1024,
        channel=3, tran0=True,
        tran1=True,
        **kwargs,
    ):
        super(
            PointNetFeat, self
        ).__init__()

        self.bneck_size = bneck_size

        self.conv1 = torch.nn.Conv1d(
            channel, 64, 1
        )
        self.conv2 = torch.nn.Conv1d(
            64, 128, 1
        )
        self.conv3 = torch.nn.Conv1d(
            128, bneck_size, 1
        )

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(
            bneck_size
        )

        self.global_feat = global_feat
        self.feature_transform = \
        feature_transform

        self.tran0 = tran0
        self.tran1 = tran1
        if self.tran0:
            self.stn = STNkd(3)
        if self.tran1:
            self.fstn = STNkd(64)

    def forward(self, x):
        n_pts = x.size()[2]

        if self.tran0:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None

        x = F.relu(
            self.bn1(self.conv1(x))
        )

        if self.tran1:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(
                x, trans_feat
            )
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(
            self.bn2(self.conv2(x))
        )
        x = self.bn3(self.conv3(x))
        x = torch.max(
            x, 2, keepdim=True
        )[0]
        x = x.view(-1, self.bneck_size)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            y = x.view(
                -1, self.bneck_size, 1
            ).repeat(1, 1, n_pts)
            return torch.cat(
                [y, pointfeat], 1
            ), trans, trans_feat, x