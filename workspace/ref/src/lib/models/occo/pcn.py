import itertools
import sys

import numpy as np
import torch
import torch.nn as nn
from lib.model.occo.pcn_util import PCNEncoder


class pcn(nn.Module):
    def __init__(self, **kwargs):
        super(pcn, self).__init__()

        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_coarse = 1024
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        # to update args, num_coarse, grid_size, grid_scale
        self.__dict__.update(kwargs)

        # 16384
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.meshgrid = [
            [-self.grid_scale, self.grid_scale, self.grid_size],
            [-self.grid_scale, self.grid_scale, self.grid_size]
        ]

        self.feat = PCNEncoder(global_feat=True, channel=3)

        # batch normalisation will destroy limit the expression
        self.folding1 = nn.Sequential(
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_coarse * 3)
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(1024+2+3, 512, 1),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1)
        )

    def build_grid(self, batch_size):
        # a simpler alternative would be: torch.meshgrid()
        # (4)
        x = np.linspace(*self.meshgrid[0])
        # (4)
        y = np.linspace(*self.meshgrid[1])
        # (16, 2)
        points = np.array(list(itertools.product(x, y)))
        # (B, 16, 2)
        points = np.repeat(
            points[np.newaxis, ...], repeats=batch_size,
            axis=0
        )
        return torch.tensor(points).float().to(self.device)

    def tile(self, tensor, multiples):
        # substitute for tf.tile:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/tile
        # Ref: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        def tile_single_axis(a, dim, n_tile):
            init_dim = a.size()[dim]
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*repeat_idx)
            order_index = torch.Tensor(
                np.concatenate(
                    [init_dim * np.arange(n_tile) + i for i in range(init_dim)]
                )
            ).long()
            return torch.index_select(a, dim, order_index.to(self.device))

        for dim, n_tile in enumerate(multiples):
            if n_tile == 1:  # increase the speed effectively
                continue
            tensor = tile_single_axis(tensor, dim, n_tile)
        return tensor

    @staticmethod
    def expand_dims(tensor, dim):
        # substitute for tf.expand_dims:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/expand_dims
        # another solution is: torch.unsqueeze(tensor, dim=dim)
        return tensor.unsqueeze(-1).transpose(-1, dim)

    def forward(self, x):
        # use the same variable naming as:
        # https://github.com/wentaoyuan/pcn/blob/master/models/pcn_cd.py
        # B X 1024
        feature = self.feat(x)

        # B X (1024 * 3)
        coarse = self.folding1(feature)
        # B X 1024 X 3
        coarse = coarse.view(-1, self.num_coarse, 3)

        # (B, 16, 2)
        grid = self.build_grid(x.shape[0])
        # (B, 16 * NC, 2)
        grid_feat = grid.repeat(1, self.num_coarse, 1)

        point_feat = self.tile(
            self.expand_dims(coarse, 2),
            [1, 1, self.grid_size ** 2, 1]
        )
        # (B, 16 * NC, 3)
        point_feat = point_feat.view([-1, self.num_fine, 3])

        # (B, 16 * NC, 1024)
        global_feat = self.tile(
            self.expand_dims(feature, 1),
            [1, self.num_fine, 1]
        )
        feat = torch.cat(
            [grid_feat, point_feat, global_feat], dim=2
        )

        center = self.tile(
            self.expand_dims(coarse, 2),
            [1, 1, self.grid_size ** 2, 1]
        )
        # (B, 16 * NC, 3)
        center = center.view([-1, self.num_fine, 3])

        fine = self.folding2(
            feat.transpose(2, 1)
        ).transpose(2, 1) + center

        return coarse, fine
