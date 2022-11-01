'''
Zhangsihao Yang
04/03/2022
ref:
https://github.com/rmccorm4/PyTorch-LMDB/blob/6b1d564596ee108c100bf346e4c538d387bc07f0/folder2lmdb.py#L23

rb = return buff
ai = another index
fp = first part
sp = second part
'''
import os
import pickle

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pyarrow as pa
import torch.utils.data as data

msgpack_numpy.patch()
assert msgpack.version >= (0, 5, 2)
MAX_MSGPACK_LEN = 1000000000

def loads(buf):
    rb = msgpack.loads(
        buf, raw=False, max_bin_len=MAX_MSGPACK_LEN,
        max_array_len=MAX_MSGPACK_LEN,
        max_map_len=MAX_MSGPACK_LEN,
        max_str_len=MAX_MSGPACK_LEN
    )
    return rb


class OcCoModelNetDataset(data.Dataset):
    def __init__(self, root, split):
        if split == 'train':
            db_path = os.path.join(
                root, f'ModelNet40_{split}_1024_middle.lmdb'
            )
        elif split == 'val':
            db_path = os.path.join(
                root, 'ModelNet40_test_1024_middle.lmdb'
            )
        self._lmdb = lmdb.open(
            db_path, subdir=os.path.isdir(db_path),
            readonly=True, lock=False, readahead=True,
            map_size=1099511627776 * 2, max_readers=100
        )
        self._txn = self._lmdb.begin(write=False)
        self._size = self._txn.stat()['entries']

        # with self.env.begin(write=False) as txn:
        self.keys = self._txn.get(b'__keys__')
        if self.keys is not None:
            self.keys = loads(self.keys)
            self._size -= 1     # delete this item
        del self._txn

    def __getitem__(self, index):
        self._txn = self._lmdb.begin(write=False)
        byteflow = self._txn.get(self.keys[index])
        unpacked = loads(byteflow)
        return unpacked

    def __len__(self):
        return self._size
        # return 64

    def __repr__(self):
        return self.__class__.__name__


def generate_another_index(index):
    fp = index // 10
    sp = index % 10
    ai = int(np.random.randint(1, 10, 1).astype(int))
    ai = fp * 10 + ((sp + ai) % 10)
    return ai


class DatasetMoCo(OcCoModelNetDataset):
    def __getitem__(self, index):
        self._txn = self._lmdb.begin(write=False)
        byteflow0 = self._txn.get(self.keys[index])
        unpacked0 = loads(byteflow0)
        # generate another index
        ai = generate_another_index(index)
        byteflow1 = self._txn.get(self.keys[ai])
        unpacked1 = loads(byteflow1)
        return unpacked0[1], unpacked1[1]
    
    # def __len__(self):
    #     return 32 * 4 * 2
