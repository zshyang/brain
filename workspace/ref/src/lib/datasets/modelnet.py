''' modelnet related dataset is inside 
this script

author:
    Zhangsihao Yang

date:
    04/22/2022

name convention:
    el = edge length
    i = intervals
    ins = indices
    pp = permutated places
'''
import os
import random

import lmdb
import numpy as np
import torch
import torch.utils.data as data
from lib.dataset.dataset import ModelNetDataset
from lib.dataset.utils import loads


def pc_jigsaw(pc, k=3, el=1.):
    ''' apply jigsaw on point cloud

    args:
        pc : point cloud
        k : number of voxels along
        each axis
        el : length of voxel (cube) 
        edge

    returns:
        pc : the permutated pc
        label : for each point
    '''
    i = [
        el * 2 / k * x - el \
        for x in np.arange(k + 1)
    ]
    assert el >= pc.__abs__().max()
    ins = np.searchsorted(
        i, pc, side='left'
    ) - 1
    label = ins[:, 0] * k ** 2 + \
    ins[:, 1] * k + ins[:, 2]

    shuffle_indices = np.arange(
        k ** 3
    )
    np.random.shuffle(shuffle_indices)
    shuffled_dict = dict()
    for i, d in enumerate(
        shuffle_indices
    ):
        shuffled_dict[i] = d

    def numberToBase(n, base=k):
        if n == 0:
            return [0]
        digits = []
        while n:
            digits.append(str(int(n % base)))
            n //= base
        return int("".join(digits[::-1]))

    for voxel_id in range(k ** 3):
        selected_points = (
            label == voxel_id
        )
        pp = shuffled_dict[voxel_id]
        loc = pp
        center_diff = np.array(
            [
                (loc // k ** 2) - (voxel_id // k ** 2),
                (loc // k ** 2) // k - (voxel_id // k ** 2) // k,
                loc % k - voxel_id % k
            ]
        ) * (2 * el)/k  # + const - edge_len
        pc[selected_points] = pc[
            selected_points
        ] + center_diff

    return pc, label


class Jigsaw(ModelNetDataset):
    def __init__(self, **kwargs):
        super(Jigsaw, self).__init__(
            **kwargs
        )

    def __getitem__(self, index):
        fn = self.fns[index]
        json_index = self.file_index[
            index
        ][0]
        inside_index = \
            self.file_index[index][1]
        point_set = self.data[
            json_index
        ][inside_index]

        if self.center:
            point_set = point_set - \
                np.expand_dims(
                    np.mean(
                        point_set, 
                        axis=0
                    ), 0
                )  # center
        if self.scale:
            dist = np.max(
                np.sqrt(
                    np.sum(
                        point_set**2, 
                        axis=1
                    )
                ), 0
            )
            point_set = point_set / \
                dist

        if self.rot_da:
            theta = np.random.uniform(
                0, np.pi * 2
            )
            rotation_matrix = \
            np.array(
                [
                    [
                        np.cos(theta),
                        -np.sin(theta)
                    ], 
                    [
                        np.sin(theta),
                        np.cos(theta)
                    ]
                ]
            )
            point_set[:, [0, 2]] = \
            point_set[:, [0, 2]].dot(
                rotation_matrix
            )
        if self.noise_da:
            point_set += \
            np.random.normal(
                0, self.noise_level,
                size=point_set.shape
            )
        
        pc, label = pc_jigsaw(
            point_set
        )
        pc = torch.from_numpy(
            pc.astype(np.float32)
        )
        y = torch.from_numpy(
            label.astype(np.int32)
        )
        return pc, y


class JigsawMoCo(Jigsaw):
    def __getitem__(self, index):
        fn = self.fns[index]
        json_index = self.file_index[
            index
        ][0]
        inside_index = \
            self.file_index[index][1]
        point_set = self.data[
            json_index
        ][inside_index]

        if self.center:
            point_set = point_set - \
                np.expand_dims(
                    np.mean(
                        point_set, 
                        axis=0
                    ), 0
                )  # center
        if self.scale:
            dist = np.max(
                np.sqrt(
                    np.sum(
                        point_set**2, 
                        axis=1
                    )
                ), 0
            )
            point_set = point_set / \
                dist

        if self.rot_da:
            theta = np.random.uniform(
                0, np.pi * 2
            )
            rotation_matrix = \
            np.array(
                [
                    [
                        np.cos(theta),
                        -np.sin(theta)
                    ], 
                    [
                        np.sin(theta),
                        np.cos(theta)
                    ]
                ]
            )
            point_set[:, [0, 2]] = \
            point_set[:, [0, 2]].dot(
                rotation_matrix
            )
        if self.noise_da:
            point_set += \
            np.random.normal(
                0, self.noise_level,
                size=point_set.shape
            )
        
        pc1, _ = pc_jigsaw(
            point_set
        )
        pc1 = torch.from_numpy(
            pc1.astype(np.float32)
        )

        pc2, _ = pc_jigsaw(
            point_set
        )
        pc2 = torch.from_numpy(
            pc2.astype(np.float32)
        )
        
        return pc1, pc2


def cspc(pc):
    minpc = pc.min(0)
    maxpc = pc.max(0)

    pc = pc - \
    (minpc + maxpc) / 2.

    dist = np.max(
        np.sqrt(
            np.sum(
                pc**2, 
                axis=1
            )
        ), 0
    )
    pc = pc / dist

    return pc


def generate_another_index(index):
    fp = index // 10
    sp = index % 10
    ai = int(np.random.randint(1, 10, 1).astype(int))
    ai = fp * 10 + ((sp + ai) % 10)
    return ai


class OcCoJigsawMoCo(
    data.Dataset
):
    '''
    name convention:
        gai = gather another index
    '''
    def __init__(
        self, root, split,
        debug, occo_ratio
    ):
        self.debug = debug
        self.occo_ratio = \
        occo_ratio
        if split == 'train':
            db_path = os.path.join(
                root,
                (
                    'ModelNet40_'
                    f'{split}_1024'
                    '_middle.lmdb'
                )
            )
        elif split == 'val':
            db_path = os.path.join(
                root, (
                    'ModelNet40_'
                    'test_1024'
                    '_middle.lmdb'
                )
            )
        self._lmdb = lmdb.open(
            db_path, 
            subdir=os.path.isdir(
                db_path
            ),
            readonly=True, 
            lock=False, 
            readahead=True,
            map_size=1099511627776*2,
            max_readers=100
        )
        self._txn = self._lmdb.begin(
            write=False
        )
        self._size = self._txn.stat()['entries']

        # with self.env.begin(write=False) as txn:
        self.keys = self._txn.get(
            b'__keys__'
        )
        if self.keys is not None:
            self.keys = loads(
                self.keys
            )
            self._size -= 1     # delete this item
        del self._txn

    def _gai(
        self, index
    ):
        '''
        name convention:
            ai = another index
        '''
        ai = int(
            np.random.randint(
                1, 
                self.__len__(), 1
            ).astype(int)
        )
        ai = (index + ai) % \
        self.__len__()
        return ai

    def _sample(self, unpacked):
        if self.occo_ratio < \
        random.uniform(0, 1):
            pc = unpacked[1]
            pc = cspc(pc)
            return pc, 0
        else:
            pc = unpacked[2]
            # random sample 
            idx = np.random.randint(
                pc.shape[0],
                size=1024
            )
            pc = pc[idx,:]
            # center and scale
            pc = cspc(pc)
            # jigsaw
            pc, _ = pc_jigsaw(pc)
            return pc, 1

    def __getitem__(self, index):
        '''
        name convention:
            ai = another index
        '''
        self._txn = self.\
        _lmdb.begin(write=False)
        byteflow = self._txn.get(
            self.keys[index]
        )
        unpacked = loads(byteflow)

        ai = generate_another_index(
            index
        )
        byteflow = self._txn.get(
            self.keys[ai]
        )
        unpacked1 = loads(byteflow)

        return self._sample(
            unpacked
        ), self._sample(
            unpacked1
        )

    def __len__(self):
        if self.debug:
            return 64
        else:
            return self._size
