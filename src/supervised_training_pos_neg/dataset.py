''' dataset for negtive and positive pair

author:
    zhangsihao yang

logs:
    20230221: file created
'''
import json
import os
import os.path as osp
from glob import glob
from typing import Dict, List

import h5py
import numpy as np
import torch
import torch.utils.data as data
from pytorch3d.structures import Meshes
from torch.autograd import Variable
from tqdm import tqdm


def load_point(point_path):
    point = []
    with open(point_path, 'r') as point_file:
        for line in point_file.readlines():
            point.append(np.fromstring(line, sep=' ')[:3])
    point = np.stack(point)
    return point


class Dataset(data.Dataset):
    ''' this is a dataset only for test
    '''

    def __init__(self, **kwargs):
        # get the data root
        data_root = kwargs.get('data_root', None)
        if data_root is None:
            raise ValueError('data_root is None')

        # load the meta info
        meta_file = kwargs.get('meta_file', None)
        if meta_file is None:
            raise ValueError('meta_file is None')

        with open(os.path.join(data_root, meta_file), 'r') as f:
            meta_info = json.load(f)

        # get the data index
        stage = kwargs.get('stage', None)
        if stage is None:
            raise ValueError('stage is None')
        if stage == 'train':
            pos_data_indexes = meta_info['train_pos']
            neg_data_indexes = meta_info['train_neg']
        elif stage == 'val':
            pos_data_indexes = meta_info['val_pos']
            neg_data_indexes = meta_info['val_neg']
        elif stage == 'test':
            pos_data_indexes = meta_info['test_pos']
            neg_data_indexes = meta_info['test_neg']
        else:
            raise ValueError('stage is not in [train, val, test]')

        # store the data
        self.lables = []
        self.inputs = []
        self.indexes = []
        processed_root = kwargs.get('processed_root', None)
        if processed_root is None:
            raise ValueError('processed_root is None')

        # debug option
        debug = kwargs.get('debug', False)
        debug_load_num = kwargs.get('debug_load_num', 10)
        if debug:
            pos_data_indexes = pos_data_indexes[
                :debug_load_num if debug_load_num < len(pos_data_indexes) else len(pos_data_indexes)]
            neg_data_indexes = neg_data_indexes[
                :debug_load_num if debug_load_num < len(neg_data_indexes) else len(neg_data_indexes)]

        # load data
        for pos_index in pos_data_indexes:
            pos_data_path = os.path.join(processed_root, f'{pos_index[0]}.npz')
            pos_data = np.load(pos_data_path)
            self.inputs.append(pos_data)
            self.lables.append(1)
            self.indexes.append(pos_index)

        for neg_index in neg_data_indexes:
            neg_data_path = os.path.join(processed_root, f'{neg_index[0]}.npz')
            neg_data = np.load(neg_data_path)
            self.inputs.append(neg_data)
            self.lables.append(0)
            self.indexes.append(neg_index)

    def __len__(self):
        return len(self.lables)

    def __getitem__(self, i):
        mesh = self.inputs[i]
        mesh_index = self.indexes[i]

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
        label = torch.tensor(self.lables[i], dtype=torch.long)

        # Dictionary for collate_batched_meshes
        collated_dict = {
            'faces': faces,
            'verts': verts,
            'ring_1': ring_1,
            'ring_2': ring_2,
            'ring_3': ring_3,
            'mesh_index': mesh_index,
            'label': label,
        }
        return collated_dict

    @staticmethod
    def __collate__(batch: List[Dict]):
        if batch is None or len(batch) == 0:
            return None

        collated_dict = {}
        for k in batch[0].keys():
            collated_dict[k] = [d[k] for d in batch]
        collated_dict['meshes'] = None

        if {'verts', 'faces'}.issubset(collated_dict.keys()):
            collated_dict['meshes'] = Meshes(
                verts=collated_dict['verts'], faces=collated_dict['faces'],)

        collated_dict.pop('verts')
        collated_dict.pop('faces')

        ring_1 = torch.stack(collated_dict['ring_1'])
        ring_2 = torch.stack(collated_dict['ring_2'])
        ring_3 = torch.stack(collated_dict['ring_3'])
        label = torch.stack(collated_dict['label'])

        meshes = collated_dict['meshes']

        # Check for empty meshes
        if meshes.isempty():
            raise ValueError("Meshes are empty.")

        # Check valid meshes equal batch size
        num_meshes = len(meshes.valid)
        # Check number of faces equal num_faces
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

        # get the label

        return {
            'verts': verts,
            'faces': faces,
            'centers': centers,
            'normals': normals,
            'ring_1': ring_1,
            'ring_2': ring_2,
            'ring_3': ring_3
        }, {
            'y': label,
        }
