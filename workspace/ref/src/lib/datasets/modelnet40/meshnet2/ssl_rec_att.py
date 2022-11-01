'''
author:
    zhangsihao yang

date:
    20220909

logs:
    20220909
        file    created
'''
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


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


class Dataset(data.Dataset):
    def __init__(self, partition='', augment='', rotation_augment=False, scale_augment=False):
        """
        Args:
            partition: train or test partition of data
            augment: type of augmentation, e.g: rotate
        """
        self.data_root = '/datasets/modelnet/meshnet2/ModelNet40/'
        self.partition = partition
        self.augment = augment
        self.rotation_augment = rotation_augment
        self.scale_augment = scale_augment

        self.category_to_idx_map = self._load_category_to_idx_map()

        self.mesh_data_dict, self.mesh_index_list = self._load_mesh_data()
        self.point_data_dict, self.point_index_list = self._load_point_data()

        # self.data = []
        # for category in os.listdir(self.data_root):
        #     category_index = self.category_to_idx_map[category]
        #     category_root = osp.join(osp.join(self.data_root, category), partition)
        #     for filename in os.listdir(category_root):
        #         if filename.endswith('.npz'):
        #             self.data.append((osp.join(category_root, filename), category_index))

    def _load_category_to_idx_map(self):
        label_root = self.data_root + '../ModelNet40.json'
        with open(label_root) as json_file:
            category_to_idx_map = json.loads(json_file.read())
        if len(category_to_idx_map) != 40:
            raise ValueError('ModelNet40 has 40 classes!')
        return category_to_idx_map

    def _load_mesh_data(self):
        mesh_data_dict = {}
        index_list = []
        for category in tqdm(os.listdir(self.data_root), desc='load mesh data'):
            category_index = self.category_to_idx_map[category]
            category_root = osp.join(osp.join(self.data_root, category), self.partition)
            for filename in os.listdir(category_root):
                if filename.endswith('.npz'):
                    # self.data.append((osp.join(category_root, filename), category_index))
                    file_index = filename.split('.')[0]
                    index_list.append(file_index)
                    file_path = osp.join(category_root, filename)
                    mesh_data_dict[file_index] = (np.load(file_path), category_index)
        return mesh_data_dict, index_list

    def _load_point_data(self):
        point_root = '/datasets/modelnet/modelnet40_ply_hdf5_2048'
        point_data_dict = {}
        index_list = []

        if self.partition == 'train':
            num_files = 5
        else:
            num_files = 2
        
        for i in tqdm(range(num_files), desc='load point data'):
            with open(osp.join(point_root, f'ply_data_{self.partition}_{i}_id2file.json'), 'r') as jf:
                id2file = json.load(jf)
            pc, label = load_h5(osp.join(point_root, f'ply_data_{self.partition}{i}.h5'))
            
            assert len(pc) == len(label) == len(id2file)

            for id2file_, pc_, label_ in zip(id2file, pc, label):
                id2file_ = id2file_.split('/')[1].split('.')[0]
                index_list.append(id2file_)
                point_data_dict[id2file_] = (pc_, label_)

        return point_data_dict, index_list

    def __getitem__(self, i):
        # Read mesh properties for .npz files
        file_index = self.point_index_list[i]
        mesh, target = self.mesh_data_dict[file_index]
        point, _ = self.point_data_dict[file_index]
        point = torch.from_numpy(point).float()
        # path, target = self.data[i]
        # mesh = np.load(path)
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
        if self.partition == 'train' and self.scale_augment:
            verts = verts.numpy()
            point = point.numpy()

            # Scale verticies during training
            for v in range(verts.shape[1]):
                verts[:, v] = verts[:, v] * np.random.normal(1, 0.1)
                point[:, v] = point[:, v] * np.random.normal(1, 0.1)

            verts = torch.from_numpy(verts).float()
            point = torch.from_numpy(point).float()

        if self.partition == 'train' and self.rotation_augment:
            # Perform rotations during training
            verts = verts.numpy()
            max_rot_ang_deg = 360
            x = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
            y = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180
            z = np.random.uniform(-max_rot_ang_deg, max_rot_ang_deg) * np.pi / 180

            A = np.array(((np.cos(x), -np.sin(x), 0),
                          (np.sin(x), np.cos(x), 0),
                          (0, 0, 1)),
                         dtype=verts.dtype)

            B = np.array(((np.cos(y), 0, -np.sin(y)),
                          (0, 1, 0),
                          (np.sin(y), 0, np.cos(y))),
                         dtype=verts.dtype)

            C = np.array(((1, 0, 0),
                          (0, np.cos(z), -np.sin(z)),
                          (0, np.sin(z), np.cos(z))),
                         dtype=verts.dtype)

            np.dot(verts, A, out=verts)
            np.dot(verts, B, out=verts)
            np.dot(verts, C, out=verts)
            verts = torch.from_numpy(verts).float()

            point = point.numpy()
            np.dot(point, A, out=point)
            np.dot(point, B, out=point)
            np.dot(point, C, out=point)
            point = torch.from_numpy(point).float()

        # Dictionary for collate_batched_meshes
        collated_dict = {
            'faces': faces,
            'verts': verts,
            'ring_1': ring_1,
            'ring_2': ring_2,
            'ring_3': ring_3,
            'target': target,
            'point': point,
            'file_index': file_index,
        }
        return collated_dict

    def __len__(self):
        return len(self.point_index_list)

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

        point = torch.stack(collated_dict['point'], 0)

        return {
            'verts': verts,
            'faces': faces,
            'centers': centers,
            'normals': normals, 
            'ring_1': ring_1, 
            'ring_2': ring_2,
            'ring_3': ring_3
        }, {
            'y': point,
            'targets': targets,
            'file_index': collated_dict['file_index'],
        }
