''' dataset object for SHREC dataset

name convention:
    nb = number of blocks

class:
    C00 SHREC16
        00 __init__(self, root, split, nb)
        01 __len__(self)
        02 __getitem__(self, index)
        03 get_cat_from_fn(fn)
            used in C00-00
        04 create_dict_cat(root)
            used in C00-00
        05 collate_fn(batch)

function:
    F00 rotate_tuple_mesh(r, tuple_mesh)
        used in C00-02

author: 
    Zhangsihao Yang

date:
    20220530

logs:
    20220531
        add rotation to the function
'''
import os
from glob import glob

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import (center_meshes, cstm, flat_meshes, load_tuple_mesh,
                    save_meshes, split_mesh, idx_to_ijk)


def rotate_tuple_mesh(r, tuple_mesh):
    v, f = tuple_mesh
    v = v @ r
    return v, f


class SHREC16(Dataset):
    def __init__(self, root, split, nb):
        ''' init of shrec 16 mesh dataset

        args:
            root: root location of the data files
            split: 'train', or 'test' split
            nb: number of block at each dimension

        name convention:
            fn = file name
            lf = list of files
            lm = list of meshes
            nb = number of blocks
            tm = tuple mesh
        '''
        self.nb = nb
        self.lf = glob(os.path.join(root, '*', split, '*.obj'))

        # create category dictionary
        dict_cat = self.create_dict_cat(root)

        # pre-compute the split mesh
        self.labels = []
        for fn in self.lf:
            # compute the cateogry and append
            cat = dict_cat[self.get_cat_from_fn(fn)]
            self.labels.append(cat)

    def __len__(self):
        return len(self.lf)

    @staticmethod
    def get_cat_from_fn(fn):
        return fn.split('/')[-3]

    @staticmethod
    def create_dict_cat(root):
        list_cat_path = glob(os.path.join(root, '*'))
        list_cat = [cat_path.split('/')[-1] for cat_path in list_cat_path]
        dict_cat = {cat: i for i, cat in enumerate(list_cat)}
        return dict_cat

    def __getitem__(self, index):
        index = index // 20 * 20
        print(index)
        # print(self.lf[10])
        # load the mesh
        tuple_mesh = load_tuple_mesh(self.lf[index])

        # rotate the mesh
        # r = R.random().as_matrix()
        # tuple_mesh = rotate_tuple_mesh(r, tuple_mesh)

        # then split the mesh into meshes
        meshes = split_mesh(cstm(tuple_mesh), self.nb)

        # move the meshes to the center 
        meshes = center_meshes(meshes, self.nb)

        # flatten the meshes, more specific flatten the faces
        meshes = flat_meshes(meshes)

        return meshes, self.labels[index], self.nb

    @staticmethod
    def __collate__(batch):
        ''' collate fucntion for shrec16 dataset
        
        args:
            batch

        name convention:
            fs: padded batch of faces
            gi: gather index
            vs: padded batch of vertices
            fms: padded batch of face masks
            vms: padded batch of vertex masks
            spidx: sparse index
        '''
        vs, vms, fs, fms, gather_index, labels, spidx = [], [], [], [], [], [], []
        spshape = []
        i = 1
        # TODO: this gather index might need to be for sparse convolution
        for batch_idx, (meshes, label, nb) in enumerate(batch):
            labels.append(torch.from_numpy(np.array(label)))
            # pcs.append(torch.from_numpy(pc))
            idxs = np.zeros((nb * nb * nb))
            for idx, tm in meshes.items():
                # Fetch.
                v, f = tm
                if v.shape[0] == 0:
                    continue
                vm = np.ones_like(v[..., 0])
                fm = np.ones_like(f)
                # Append.
                vs.append(torch.from_numpy(v))
                vms.append(torch.from_numpy(vm))
                fs.append(torch.from_numpy(f))
                fms.append(torch.from_numpy(fm))
                # idxs.append(idx)
                idxs[idx] = i
                i = i + 1

                ijk = idx_to_ijk(idx, nb)
                # print(ijk)

                spidx.append([batch_idx, ijk[0], ijk[1], ijk[2]])

            spshape.append(i)

            gather_index.append(idxs)
        # Pad.
        vs = pad_sequence(vs, batch_first=True)
        vms = 1- pad_sequence(vms, batch_first=True)
        fs = pad_sequence(fs, batch_first=True)
        fms = 1 - pad_sequence(fms, batch_first=True)
        gi = torch.from_numpy(np.stack(gather_index)).type(torch.int64)
        y = torch.stack(labels, 0)

        spidx = torch.from_numpy(np.array(spidx)).type(torch.int32)
        spshape = [nb, nb, nb]

        return {
            'vs': vs, 'vms': vms, 'fs': fs, 'fms': fms,
            'gi': gi, 'spidx': spidx, 'spshape': spshape,
            'batch_size': len(batch)
        }, {'y': y}
