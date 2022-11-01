'''
class:
    C00 SHREC16(base)
        00 __getitem__(self, index)
            <-- F00
        01 __collate__(batch)

function:
    f00 process_points(point_set)
    f01 rotate_points(verts)

author:
    zhangsihao yang

date:
    20220820

logs:
    20220820
        created
    20220821
        modify C00-01
    20220824
    20220830
        f01     added
'''
import numpy as np
import trimesh
import torch
from lib.dataset.mesh.shrec_basic import base


def process_points(point_set):
    point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0) # center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    point_set = point_set / dist #scale
    return point_set


def rotate_points(verts):
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

    return verts


class SHREC16(base):
    def __init__(self, root, split, augment=None):
        super(SHREC16, self).__init__(root, split)
        self.augment = augment

    def __getitem__(self, index):
        tmesh = trimesh.load(self.lf[index], process=False)
        sampled_points, _ = trimesh.sample.sample_surface(tmesh, 1024)
        sampled_points = process_points(sampled_points)
        sampled_points = np.array(sampled_points, dtype=np.float32)

        if self.augment == 'rotate':
            sampled_points = rotate_points(sampled_points)

        return sampled_points, self.labels[index]

    @staticmethod
    def __collate__(batch):
        num_voxel = 256

        batch_size = len(batch)
        assert batch_size > 0, 'batch size must be a positive integer'

        coords = [batch[i][0] for i in range(len(batch))]
        labels = [batch[i][1] for i in range(len(batch))]
        labels = torch.tensor(labels)

        # concat the feature
        features = np.vstack(coords).astype(np.float32)

        # create indices
        num_points = features.shape[0] // batch_size
        indices = [i for i in range(batch_size) for _ in range(num_points)]
        indices = np.array(indices, dtype=np.int32)
        xyz = (features + 1.) * num_voxel // 2
        indices = np.concatenate((np.expand_dims(indices, -1), xyz.astype(np.int32)), axis=-1)

        spshape = [num_voxel, num_voxel, num_voxel]

        # convert numpy array to torch tensor
        features = torch.from_numpy(features)
        indices = torch.from_numpy(indices)

        return {
            'x': {
                'features': features, 'indices': indices,
                'spshape': spshape, 'batch_size': batch_size
            }
        }, {
            'y': labels
        }
