'''
author:
    zhangsihao yang

logs:
    20220918
        file created
'''
import os
from glob import glob

from tqdm import tqdm
import json
import os
import os.path as osp
from typing import Dict, List

import h5py
import numpy as np
import torch
import torch.utils.data as data
from pytorch3d.structures import Meshes
from torch.autograd import Variable
from tqdm import tqdm


class Dataset(data.Dataset):
    ''' this is a dataset only for test
    '''
    def __init__(self, **kwargs):
        self.data_root = '/datasets/shapenet/part/mesh/aligned'

        list_npz_path= glob(
            os.path.join(self.data_root, '*/*/aligned.npz')
        )

        self.mesh_data_dict = {}
        self.list_mesh_index = []

        for npz_path in tqdm(list_npz_path, desc='load mesh data'):
            mesh_index = npz_path.split('/')
            mesh_index = '/'.join([mesh_index[-3], mesh_index[-2]])
            self.list_mesh_index.append(mesh_index)
            self.mesh_data_dict[mesh_index] = np.load(npz_path)

    def __len__(self):
        return len(self.list_mesh_index)

    def __getitem__(self, i):
        # Read mesh properties for .npz files
        mesh_index = self.list_mesh_index[i]
        mesh = self.mesh_data_dict[mesh_index]

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

        # Dictionary for collate_batched_meshes
        collated_dict = {
            'faces': faces,
            'verts': verts,
            'ring_1': ring_1,
            'ring_2': ring_2,
            'ring_3': ring_3,
            'mesh_index': mesh_index
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
                verts=collated_dict['verts'],
                faces=collated_dict['faces'],
            )

        collated_dict.pop('verts')
        collated_dict.pop('faces')

        ring_1 = torch.stack(collated_dict['ring_1'])
        ring_2 = torch.stack(collated_dict['ring_2'])
        ring_3 = torch.stack(collated_dict['ring_3'])

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

        return {
            'verts': verts,
            'faces': faces,
            'centers': centers,
            'normals': normals,
            'ring_1': ring_1,
            'ring_2': ring_2,
            'ring_3': ring_3
        }, {
            'mesh_index': collated_dict['mesh_index']
        }
