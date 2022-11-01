'''
class:
    C00 base
        00 __init__(self, root, split, **kwargs)
            <-- 01
            <-- 02
        01 create_dict_cat(root)
        02 get_cat_from_fn(fn)
        03 __len__(self)

author:
    zhangsihao yang

date:
    20220907

logs:
    20220907
        file    created
'''
import os
from glob import glob

from torch.utils.data import Dataset


class base(Dataset):
    def __init__(self, root, split):
        ''' init of shrec 16 mesh dataset

        args:
            root: root location of the data files
            split: 'train', or 'test' split

        name convention:
            fn = file name
            lf = list of files
        '''
        # get the list of files
        self.lf = glob(os.path.join(root, '*', split, '*.obj'))

        # create category dictionary
        self.dict_cat = self.create_dict_cat(root)

        # get the list of label index
        self.labels = []
        for fn in self.lf:
            # compute the cateogry and append
            cat = self.dict_cat[self.get_cat_from_fn(fn)]
            self.labels.append(cat)

    @staticmethod
    def create_dict_cat(root):
        # each category is projected to a number
        list_cat_path = glob(os.path.join(root, '*'))
        # convert to a list with strings of category
        list_cat = [cat_path.split('/')[-1] for cat_path in list_cat_path]
        # convert to a dictionary
        dict_cat = {cat: i for i, cat in enumerate(list_cat)}
        return dict_cat

    @staticmethod
    def get_cat_from_fn(fn):
        return fn.split('/')[-3]

    def __len__(self):
        return len(self.lf)



""" Data loader for SHREC11 data set

class:
    c00 SHREC11(data.Dataset)
        00  __init__(self, data_root, partition, category_to_idx_map, augment)
        01 __getitem__(self, i)
        02 __len__(self)
        03 __collate__(batch])

ref:
    meshnet2
"""
import json
import os
import os.path as osp
from typing import Dict, List

import numpy as np
import torch
import torch.utils.data as data
from pytorch3d.structures import Meshes
from torch.autograd import Variable


class ModelNet40(data.Dataset):
    """ ModelNet40 dataset """
    def __init__(self, data_root='', partition='', augment=''):
        """
        Args:
            data_root: root directory where the SHREC11 dataset is stored
            partition: train or test partition of data
            augment: type of augmentation, e.g: rotate
        """
        self.data_root = data_root
        self.partition = partition

        label_root = data_root + '../ModelNet40.json'
        with open(label_root) as json_file:
            category_to_idx_map = json.loads(json_file.read())
        if len(category_to_idx_map) != 40:
            raise ValueError('ModelNet40 has 40 classes!')

        self.category_to_idx_map = category_to_idx_map
        self.augment = augment
        self.data = []
        for category in os.listdir(self.data_root):
            category_index = self.category_to_idx_map[category]
            category_root = osp.join(osp.join(self.data_root, category), partition)
            for filename in os.listdir(category_root):
                if filename.endswith('.npz'):
                    self.data.append((osp.join(category_root, filename), category_index))

    def __getitem__(self, i):
        # Read mesh properties for .npz files
        path, target = self.data[i]
        mesh = np.load(path)
        faces = mesh['faces']
        verts = mesh['verts']
        ring_1 = mesh['ring_1']
        ring_2 = mesh['ring_2']
        ring_3 = mesh['ring_3']

        # Convert to tensor
        faces = torch.from_numpy(faces).long()
        verts = torch.from_numpy(verts).float()
        ring_1 = torch.from_numpy(ring_1).long()
        ring_2 = torch.from_numpy(ring_2).long()
        ring_3 = torch.from_numpy(ring_3).long()
        target = torch.tensor(target, dtype=torch.long)

        # Perform augmentation during training
        if self.partition == 'train' and self.augment == 'scale_verts':
            verts = verts.numpy()
            # Scale verticies during training
            for v in range(verts.shape[1]):
                verts[:, v] = verts[:, v] * np.random.normal(1, 0.1)

            verts = torch.from_numpy(verts).float()

        # Dictionary for collate_batched_meshes
        collated_dict = {
            'faces': faces,
            'verts': verts,
            'ring_1': ring_1,
            'ring_2': ring_2,
            'ring_3': ring_3,
            'target': target
        }
        return collated_dict

    def __len__(self):
        return len(self.data)

    @staticmethod
    def __collate__(batch: List[Dict]):
        """
        Take a list of objects in the form of dictionaries and merge them
        into a single dictionary. This function is used with a Dataset
        object to create a torch.utils.data.Dataloader which directly
        returns Meshes objects.

        Args:
            batch: List of dictionaries containing information about objects
            in the dataset.

        Returns:
            collated_dict: Dictionary of collated lists. If batch contains both
            verts and faces, a collated mesh batch is also returned.
        """
        if batch is None or len(batch) == 0:
            return None
        collated_dict = {}
        for k in batch[0].keys():
            collated_dict[k] = [d[k] for d in batch]
        collated_dict['meshes'] = None

        if {'verts', 'faces'}.issubset(collated_dict.keys()):
            collated_dict['meshes'] = Meshes(
                verts=collated_dict['verts'],
                faces=collated_dict['faces'],
            )

        collated_dict.pop('verts')
        collated_dict.pop('faces')

        ring_1 = torch.stack(collated_dict['ring_1'])
        ring_2 = torch.stack(collated_dict['ring_2'])
        ring_3 = torch.stack(collated_dict['ring_3'])
        targets = torch.stack(collated_dict['target'])

        meshes = collated_dict['meshes']

        #Check for empty meshes
        if meshes.isempty():
            raise ValueError("Meshes are empty.")
        #Check valid meshes equal batch size
        num_meshes = len(meshes.valid)
        #Check number of faces equal num_faces
        num_faces = meshes.num_faces_per_mesh().max().item()
        # Each vertex is a point with x,y and z co-ordinates
        verts = meshes.verts_padded()
        # Normals for scaled vertices
        normals = meshes.faces_normals_padded()
        # Each face contains index of its corner vertex
        faces = meshes.faces_padded()

        if not torch.isfinite(verts).all():
            raise ValueError("Mesh vertices contain nan or inf.")
        if not torch.isfinite(normals).all():
            raise ValueError("Mesh normals contain nan or inf.")

        corners = verts[torch.arange(num_meshes)[:, None, None], faces.long()]
        centers = torch.sum(corners, axis=2)/3
        # Each mesh face has one center
        assert centers.shape == (num_meshes, num_faces, 3)
        # Each face only has 3 corners
        assert corners.shape == (num_meshes, num_faces, 3, 3)

        assert ring_1.shape == (num_meshes, num_faces, 3)

        assert ring_2.shape == (num_meshes, num_faces, 6)

        assert ring_3.shape == (num_meshes, num_faces, 12)

        centers = centers.permute(0, 2, 1)
        normals = normals.permute(0, 2, 1)

        verts = Variable(verts)
        faces = Variable(faces)
        centers = Variable(centers)
        normals = Variable(normals)

        ring_1 = Variable(ring_1)
        ring_2 = Variable(ring_2)
        ring_3 = Variable(ring_3)

        targets = Variable(targets.cuda())

        return {
            'verts': verts,
            'faces': faces,
            'centers': centers,
            'normals': normals, 
            'ring_1': ring_1, 
            'ring_2': ring_2,
            'ring_3': ring_3
        }, {
            'y': targets
        }
