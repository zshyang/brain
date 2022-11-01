'''
Zhangsihao Yang
04/12/2022

bel = box edge length
csv = center scale vertices
cn = class name
f = faces
fc = face centers
fn = file name
fns = file name split
lm = list of meshes
m = meshes
mnb = maximum number of blocks
mnf = maximum number of faces
nb = number of blocks
pp = post processing
rd = random drop
tm = tuple mesh
v = vertices

functions:
    save_mesh(v, f, fn)
'''
import os
import pickle
import random

import lmdb
import numpy as np
import pyarrow as pa
import trimesh
from lib.dataset.utils import load_ply
from torch.utils.data import Dataset


def save_mesh(v, f, fn):
    ''' save the mesh as an obj file

    Args:
        v: vertices
        f: faces
        fn: save file name

    Name convention:
        mf = mesh file
    '''
    # f starts with index 0 while obj file requires index starts at 1
    f = f.astype(np.int32) + 1
    with open(fn, 'w') as mf:
        # write vertices into file
        for vertex in v:
            mf.write(f'v {vertex[0]:f} {vertex[1]:f} {vertex[2]:f}\n')
        # write faces into file
        for face in f:
            mf.write(f'f {face[0]:d} {face[1]:d} {face[2]:d}\n')


def create_space_points(split):
    ''' create a even spaced voxel point centers given 
    the split parameters

    Args:
        split: The tuple contains the split parameters
    
    Returns:
        space_points: the voxel centers
    '''
    # A corner case.
    if split is None:
        return None

    num_x, num_y, num_z = split

    # The 3D object is in a unit sphere. So the minimum 
    # and maximum in each axis is 0.5.
    space_min = -0.5
    space_max = 0.5
    half_x = 1.0 / num_x * 0.5
    half_y = 1.0 / num_y * 0.5
    half_z = 1.0 / num_z * 0.5

    # Compute the sample point on each axis.
    x_sample = np.linspace(space_min, space_max, num=num_x+1)
    x_sample = x_sample[0:-1] + half_x
    y_sample = np.linspace(space_min, space_max, num=num_y+1)
    y_sample = y_sample[0:-1] + half_y
    z_sample = np.linspace(space_min, space_max, num=num_z+1)
    z_sample = z_sample[0:-1] + half_z

    # Add the sample points together.
    space_points = np.zeros((num_x, num_y, num_z, 3))
    # this will make space_points[i, :, :, 0] the same
    space_points[..., 0] = space_points[..., 0] + x_sample.reshape(-1, 1, 1)
    # this will make space_points[:, j, :, 1] the same
    space_points[..., 1] = space_points[..., 1] + y_sample.reshape(1, -1, 1)
    # this will make space_points[:, :, k, 2] the same
    space_points[..., 2] = space_points[..., 2] + z_sample.reshape(1, 1, -1)
    space_points = np.reshape(space_points, (-1, 3))

    return space_points


def get_face_centers(tm):
    ''' get the centers of each face of the input mesh

    Args:
        tm: tuple mesh represents
            [0]: vertices
            [1]: faces

    Returns:
        fc: a numpy array with centers of the faces
    '''
    v, f = tm
    fc = v[f].mean(axis=1)
    return fc


def distance_between_two_point_clouds(pc1, pc2):
    ''' compute the distance between two point clouds. 
    this function takes the reference from:
    https://github.com/WangYueFt/dgcnn/blob/e96a7e26555c3212dbc5df8d8875f07228e1ccc2/tensorflow/utils/tf_util.py#L638
    note that the distance is squared distance

    Args:
        pc1: A point cloud with shape (n1, 3).
        pc2: A point cloud with shape (n2, 3).

    Returns:
        dist: A distance map with shape (n1, n2).
    '''
    # Prepare the size.
    pc1 = np.reshape(pc1, (-1, 3))
    pc2 = np.reshape(pc2, (-1, 3))

    # Compute the components to get the distance.
    pc_inner = -2. * pc1 @ pc2.T
    pc1_square = np.square(pc1).sum(-1, keepdims=True)
    pc2_square_t = np.square(pc2).sum(
        -1, keepdims=True
    ).T

    # Add up the components
    dist = pc1_square + pc_inner + pc2_square_t

    return np.sqrt(dist)


def clean_order_mesh(vertices, group_faces):
    """Reorder the vertices and faces in order to remove some useless vertices.
    Clean the mesh and also reorder the mesh.

    Becuase there is only part of the vertices is useful. Remove some unused 
    vertices.

    Args:
        vertices: The numpy array of vertices.
        group_faces: The group of faces.
    
    Returns:
        vertices: The ordered vertices. It is a numpy array.
        faces: The ordered faces. It is a list.
    """
    # After removing degenerate faces some vertices are now unreferenced.
    # Remove these.
    if len(group_faces) == 0:
        return [], []
    else:
        num_verts = vertices.shape[0]  # get the number of vertices
        # Get the vertices that are connected by the faces.
        vert_connected = np.equal(
            np.arange(num_verts)[:, None],
            np.hstack(group_faces)[None]).any(axis=-1)  # (Nv,)
        # The new vertices.
        vertices = vertices[vert_connected]
        vert_indices = (
                np.arange(num_verts) - np.cumsum(
                    1 - vert_connected.astype('int')))
        vert_indices = np.cumsum(vert_connected) - 1
        faces = [vert_indices[f].tolist() for f in group_faces]

        return vertices, faces


def xyz_to_ijk_floor(xyz, bel):
    ''' convert from float to floor integer.
    pick the conner with smallest ijk.

    figure below is an example with nb = 4, bel = 0.25
    a = -0.5, b = -0.25, c = 0.0, d = 0.25, e = 0.5
    a// = -2, b// = -1, c// = 0, d// = 1, e// = 2
    a     b     c     d     e
    .--|--.--|--.--|--.--|--.
       0     1     2     3
    
    first step is to translate float number
    a = 0.0, b = 0.25, c = 0.5, d = 0.75, e = 1.0
    a// = 0, b// = 1, c// = 2, d// = 3, e// = 4
    a     b     c     d     e
    .--|--.--|--.--|--.--|--.
       0     1     2     3
    
    second step is to shift a, b, c, d, e by --
    a = -0.125, b = 0.125, c = 0.375, d = 0.625, e = 0.825
    a// = -1, b// = 0, c// = 1, d// = 2, e// = 3
    a     b     c     d     e
       .--|--.--|--.--|--.--|--.
       0     1     2     3
    '''
    ijk = (xyz + 0.5 - 0.5 * bel) // bel
    ijk = ijk * (ijk >= 0)
    return ijk.astype(np.int32)


def xyz_to_ijk(xyz, bel):
    ''' convert from float number to integer

    figure below is an example with nb = 3, bel = 0.25
    a = -0.5, b = -0.25, c = 0.0, d = 0.25, e = 0.5
    a     b     c     d     e
    .--|--.--|--.--|--.--|--.
       0     1     2     3

    first step is to translate float number
    a = 0.0, b = 0.25, c = 0.5, d = 0.75, e = 1.0
    a     b     c     d     e
    .--|--.--|--.--|--.--|--.
       0     1     2     3

    second step is to add -- to a, b, c, d, e
    a = a--, b = b--, c = c--, d = d--, e = e--
       a     b     c     d     e
    .--|--.--|--.--|--.--|--.
       0     1     2     3
    '''
    ijk = (xyz + 0.5 + 0.5 * bel) // bel
    return ijk


def ijk_to_xyz(ijk, bel):
    ''' convert from integer to float number

    figure below is an example with nb = 4, bel = 0.25
    a = -0.5, b = -0.25, c = 0.0, d = 0.25, e = 0.5
       0     1     2     3
    .--|--.--|--.--|--.--|--.
    a     b     c     d     e
    '''
    xyz = 0.5 * bel + ijk * bel - 0.5
    return xyz    


def ijk_to_ijkn8(ijk):
    ''' find the ijk of 8 upper neighborings of ijk.

    Args:
        ijk: with shape (n, 3)

    Returns:
        ijkn8: with shape (n, 8, 3)
    '''
    z = np.zeros((ijk.shape[0], 2, 2, 2, 3))
    ijk = ijk.reshape(-1, 1, 1, 1, 3)
    step = np.array([0, 1], dtype=np.int32)
    z[...,0] = ijk[...,0] + step.reshape(1, 2, 1, 1)
    z[...,1] = ijk[...,1] + step.reshape(1, 1, 2, 1)
    z[...,2] = ijk[...,2] + step.reshape(1, 1, 1, 2)
    ijkn8 = z.reshape(-1, 8, 3)
    return ijkn8


def ijk_to_idx(ijk, nb):
    idx = ijk[..., 0] * nb * nb + \
        ijk[..., 1] * nb + \
        ijk[..., 2]
    return idx


def convert_fc_to_idx(fc, bel, nb):
    ''' idx is an important concept. it is an array.
    each row has 8 numbers. their absolute value is the 
    index of 8 neighborings of the center of a face.
    if the value is a positive number, it is a valid
    index. otherwise, it is an invalid index.
    '''
    # (n, 3)
    ijk = xyz_to_ijk_floor(fc, bel)
    # (n, 8, 3)
    ijkn8 = ijk_to_ijkn8(ijk)
    xyzn8 = ijk_to_xyz(ijkn8, bel)
    # (n, 1, 3)
    fc = fc.reshape(-1, 1, 3)
    # (n, 8)
    dist = np.sqrt(((fc - xyzn8)**2).sum(-1))
    flag = ((dist <= (bel * 0.87)) * 2 - 1)

    idx = ijk_to_idx(ijkn8, nb)
    idx = (idx + 1) * flag - 1
    return idx.astype(np.int32)


def convert_idx_to_meshes(tuple_mesh, idx):
    ''' convert mesh into meshes according to index

    Args:
        tuple_mesh: the input mesh
        index: index use to split the mesh

    Returns:
        meshes: the dict of meshes

    Name convetion:
        ui = unique index
    '''
    meshes = {}
    v, f = tuple_mesh
    ui = np.unique(idx)
    ui = ui[ui >= 0]
    for group_idx in ui:
        a, b = np.where(idx == group_idx)
        group_mesh = clean_order_mesh(v, f[a])
        meshes[group_idx] = (
            group_mesh[0], 
            np.array(group_mesh[1], np.int32)
        )
    return meshes


def split_mesh(tm, nb):
    ''' split tuple mesh given split information

    Args:
        tm: a tuple with
            [0]: numpy array of vertices
            [1]: numpy array of faces
        nb: number of blocks along each axis

    Returns:
        meshes: a dict of tuple meshes
    '''
    fc = get_face_centers(tm)
    bel = 1. / nb
    idx = convert_fc_to_idx(fc, bel, nb)
    meshes = convert_idx_to_meshes(tm, idx)
    return meshes


def save_meshes(folder, meshes):
    '''
    Usage:
        save_meshes(
            unpacked[0].replace('/', '_').split('.')[0],
            meshes
        )
    '''
    os.makedirs(folder, exist_ok=True)
    for idx, tuple_mesh in meshes.items():
        v, f = tuple_mesh
        fn = os.path.join(folder, f'{idx}.obj')
        save_mesh(v, f, fn)


def load_tuple_mesh_as_trimesh(tuple_mesh):
    v, f = tuple_mesh
    tri_mesh = trimesh.Trimesh(
        vertices=v, faces=f
    )
    return tri_mesh


def sim_tuple_mesh(tuple_mesh, mnf):
    tri_mesh = load_tuple_mesh_as_trimesh(tuple_mesh)
    sim_mesh = tri_mesh.simplify_quadratic_decimation(
        mnf
    )
    v = np.array(sim_mesh.vertices, np.float32)
    f = np.array(sim_mesh.faces, np.int32)
    return v, f


def simplify_meshes(meshes, mnf):
    for key, tm in list(meshes.items()):
        v, f = tm
        if f.shape[0] > mnf:
            tm = sim_tuple_mesh(
                tm, mnf
            )
            if tm[1].shape[0] > mnf:
                del meshes[key]
            else:
                meshes[key] = tm
    return meshes


def idx_to_ijk(idx, nb):
    i = (idx // (nb * nb)).astype(np.int32)
    j = ((idx - i * nb * nb) // nb).astype(np.int32)
    k = idx - i * nb * nb - j * nb
    return np.stack((i, j, k))


def unit_cube():
    v = np.array(
        [
            [0.0,  0.0,  0.0],
            [0.0,  0.0,  1.0],
            [0.0,  1.0,  0.0],
            [0.0,  1.0,  1.0],
            [1.0,  0.0,  0.0],
            [1.0,  0.0,  1.0],
            [1.0,  1.0,  0.0],
            [1.0,  1.0,  1.0],
        ]
    )
    f = np.array(
        [
            [1, 7, 5],
            [1, 3, 7], 
            [1, 4, 3], 
            [1, 2, 4], 
            [3, 8, 7], 
            [3, 4, 8], 
            [5, 7, 8], 
            [5, 8, 6], 
            [1, 5, 6], 
            [1, 6, 2], 
            [2, 6, 8], 
            [2, 8, 4], 
        ]
    )
    return v, f


def save_meshes_as_voxel(fldr, meshes, nb):
    ''' save meshes as a voxel

    Args:
        fldr: the folder to save the voxel mesh.
        meshes: a dict with key as the group index.
            value as a tuple mesh.
        nb: integer

    Name convention:
        nb = number of blocks
    
    Usage:
        save_meshes_as_voxel(
            unpacked[0].replace('/', '_').split('.')[0],
            meshes, self.nb
        )
    '''
    os.makedirs(fldr, exist_ok=True)
    vertices = []
    faces = []
    gap = 0
    for key in meshes:
        ijk = idx_to_ijk(key, nb)
        v, f = unit_cube()
        v = v + ijk
        f = f - 1
        vertices.append(v)
        faces.append(f + gap)
        gap += 8
    vertices = np.concatenate(vertices)
    faces = np.concatenate(faces)
    voxelfn = os.path.join(fldr, 'voxel.obj')
    save_mesh(vertices, faces, voxelfn)
    

def csv(v, s=0.468):
    # center
    v = v - np.expand_dims(
        (v.max(0) + v.min(0)) / 2. , 0
    )
    # scale
    dist = np.max(np.sqrt(
        np.sum(v ** 2, axis=1)), 0
    )
    v = v / dist * s
    return v


def center_meshes(meshes, nb):
    for idx, tm in meshes.items():
        xyz = ijk_to_xyz(idx_to_ijk(idx, nb), 1./nb)
        v, f = tm
        v = v - xyz
        meshes[idx] = (v.astype(np.float32), f)
    return meshes


def flat_f(f):
    '''Converts from list of faces to flat face array 
    with stopping indices.
    [-1] is the stop sign of a face.
    [-2] is the stop sign of all faces.
    And all this stop sign has been added 2 in the end.

    args:
        f

    returns:
        The flatten faces
    '''
    if not f.any():  # faces is empty
        return np.array([0])
    else:
        n = f.shape[0]
        sign = np.zeros((n, 1), dtype=np.int32)
        sign = sign -1
        sign[-1] = -2
        f = np.concatenate((f, sign), 1).reshape(-1) + 2
        return f


def flat_meshes(meshes):
    for idx, tm in meshes.items():
        v, f = tm
        f = flat_f(f)
        meshes[idx] = (v, f)
    return meshes


def merge_meshes(meshes):
    v = []
    f = []
    len_v = 0
    for idx, tm in meshes.items():
        v.append(tm[0])
        f.append(tm[1] + len_v)
        len_v += tm[0].shape[0]
    v = np.concatenate(v)
    f = np.concatenate(f)
    return v, f


class ShapeNet642(Dataset):
    def __init__(self, split, mnb):
        self.mnb = mnb
        base = os.path.join(
            '/dataset', 'shapenet', 
            'voxsim', 'lmdb'
        )
        if split == 'train':
            db_path = os.path.join(
                base,
                'train_64_2.lmdb'
            )
        elif split == 'val':
            db_path = os.path.join(
                base,
                'val_64_2.lmdb'
            )
        self._load_lmdb(db_path)

    def _load_lmdb(self, db_path):
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
        keys = self._txn.get(
            b'__keys__'
        )
        if keys is not None:
            self.keys = pickle.loads(
                keys
            )
        self.length = pickle.loads(
            self._txn.get(b'__len__')
        )
        del self._txn

    def __len__(self):
        return 16
        # return self.length - 1

    def _pp(self, unpacked):
        unpacked[0] = self._lm2m(
            unpacked[0]
        )
        self._rd(unpacked[0])
        unpacked[1] = np.array(
            unpacked[1], 
            np.float32
        )
        return unpacked[:3]

    def _rd(self, meshes):
        if len(meshes) > self.mnb:
            for i in range(
                len(meshes) - self.mnb
            ):
                meshes.pop(
                    random.choice(
                        list(meshes.keys())
                    )
                )

    @staticmethod
    def _lm2m(lm):
        m = {}
        for idx, tm in lm.items():
            v, f = tm
            m[int(idx)] = (
                np.array(
                    v, np.float32
                ),
                np.array(
                    f, np.int32
                )
            )
        return m

    def __getitem__(self, index):
        self._txn = self._lmdb.begin(
            write=False
        )
        byteflow = self._txn.get(
            self.keys[index]
        )
        unpacked = pickle.loads(
            byteflow
        )
        unpacked = self._pp(unpacked)
        return unpacked


class ShapeNet(Dataset):
    def __init__(self, root, split, nb, mnf):
        self.root = root
        self.nb = nb
        self.mnf = mnf

        if split == 'train':
            db_path = os.path.join(
                root, 'mansim', 'lmdb', 'train.lmdb'
            )
        elif split == 'val':
            db_path = os.path.join(
                root, 'mansim', 'lmdb', 'val.lmdb'
            )

        self._lmdb = lmdb.open(
            db_path, subdir=os.path.isdir(db_path),
            readonly=True, lock=False, readahead=True,
            map_size=1099511627776 * 2, max_readers=100
        )
        self._txn = self._lmdb.begin(write=False)
        # self._size = self._txn.stat()['entries']

        # with self.env.begin(write=False) as txn:
        self.keys = self._txn.get(b'__keys__')
        if self.keys is not None:
            self.keys = pa.deserialize(self.keys)
            self.length = pa.deserialize(
                self._txn.get(b'__len__')
            )
        del self._txn

    def _parse_cls_idx(self, fn):
        fns = fn.split('/')
        cn = fns[6]
        idx = fns[7]
        return cn, idx

    def _group_pcfn(self, cls_idx):
        pcfn = os.path.join(
            self.root, 
            'shape_net_core_uniform_samples_2048',
            'data' , cls_idx[0], f'{cls_idx[1]}.ply'
        )
        return pcfn

    def __getitem__(self, index):
        self._txn = self._lmdb.begin(write=False)

        byteflow = self._txn.get(self.keys[index])
        unpacked = pa.deserialize(
            byteflow
        )
        if self.nb < 0:
            return unpacked[0]

        v = unpacked[1]
        v = csv(v).astype(np.float32)

        tm = (v, unpacked[2].astype(np.int32))

        # ========= compute the split meshes ========
        meshes = split_mesh(tm, self.nb)
        meshes = simplify_meshes(meshes, self.mnf)
        # save_meshes(
        #     unpacked[0].replace('/', '_').split('.')[0],
        #     meshes
        # )
        # v, f = merge_meshes(meshes)
        # save_mesh(
        #     v, f, 
        #     unpacked[0].replace('/', '_'),
        # )
        meshes = center_meshes(meshes, self.nb)
        meshes = flat_meshes(meshes)

        # ========= load the point cloud ========
        cls_idx = self._parse_cls_idx(unpacked[0])
        pcfn = self._group_pcfn(cls_idx)
        if os.path.exists(pcfn):
            pc = load_ply(pcfn)
        else:
            pc = v[
                np.random.randint(
                    0, v.shape[0], size=2048
                )
            ]

        return meshes, pc, self.nb, unpacked[0]

    def __len__(self):
        return self.length
        # return 64

    def __repr__(self):
        return self.__class__.__name__
