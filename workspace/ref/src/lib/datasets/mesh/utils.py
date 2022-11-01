''' utility function for mesh datasets

name convention:
    fn = file name
    tm = tuple mesh
    bel = box edge length
    csv = center scale vertices
    cstm = center and scale tuple mesh

class:
    xxxx

function:
    F01 center_meshes(meshes, nb)
    F02 idx_to_ijk(idx, nb)
    F03 flat_meshes(meshes)
    F04 flat_f(f)
    F05 save_meshes(folder, meshes)
    F06 cstm(tm)
    F07 csv(v, s)
    F08 get_face_centers(tm)
    F09 clean_order_mesh(vertices, group_faces)
    F10 xyz_to_ijk_floor(xyz, bel)
    F11 xyz_to_ijk(xyz, bel)
    F12 ijk_to_xyz(ijk, bel)
    F13 ijk_to_ijkn8(ijk)
    F14 ijk_to_idx(ijk, nb)
    F15 convert_fc_to_idx(fc, bel, nb)
    F16 convert_idx_to_meshes(tuple_mesh, idx)
    F17 load_tuple_mesh(fn)
    F18 split_mesh(tm, nb)

author: 
    Zhangsihao Yang

date:
    20220530

log:
    20220530
        create function load_tuple_mesh
        copy functions from .mesh_dataset.py
    20220531
        copy more functions from .mesh_dataset.py
'''
import os

import numpy as np
import trimesh


def center_meshes(meshes, nb):
    for idx, tm in meshes.items():
        xyz = ijk_to_xyz(idx_to_ijk(idx, nb), 1./nb)
        v, f = tm
        v = v - xyz
        meshes[idx] = (v.astype(np.float32), f)
    return meshes


def idx_to_ijk(idx, nb):
    i = (idx // (nb * nb)).astype(np.int32)
    j = ((idx - i * nb * nb) // nb).astype(np.int32)
    k = idx - i * nb * nb - j * nb
    return np.stack((i, j, k))


def flat_meshes(meshes):
    for idx, tm in meshes.items():
        v, f = tm
        f = flat_f(f)
        meshes[idx] = (v, f)
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


def cstm(tm):
    v, f = tm
    v = csv(v)
    return v, f


def csv(v, s=0.498):
    # center according to bounding box
    v = v - np.expand_dims((v.max(0) + v.min(0)) / 2. , 0)
    # longest distance
    dist = np.max(np.sqrt(np.sum(v ** 2, axis=1)), 0)
    # scale the mesh
    v = v / dist * s
    return v


def get_face_centers(tm):
    ''' get the centers of each face of the input mesh

    args:
        tm: tuple mesh represents
            [0]: vertices
            [1]: faces

    returns:
        fc: a numpy array with centers of the faces

    name convention:
        fc = face centers
    '''
    v, f = tm
    fc = v[f].mean(axis=1)
    return fc


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
        vert_indices = (np.arange(num_verts) - np.cumsum(1 - vert_connected.astype('int')))
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


def ijk_to_ijkn8(ijk, nb):
    ''' find the ijk of 8 upper neighborings of ijk.

    args:
        ijk: with shape (n, 3)

    returns:
        ijkn8: with shape (n, 8, 3)
    '''
    z = np.zeros((ijk.shape[0], 2, 2, 2, 3))
    ijk = ijk.reshape(-1, 1, 1, 1, 3)
    step = np.array([0, 1], dtype=np.int32)
    z[...,0] = ijk[...,0] + step.reshape(1, 2, 1, 1)
    z[...,1] = ijk[...,1] + step.reshape(1, 1, 2, 1)
    z[...,2] = ijk[...,2] + step.reshape(1, 1, 1, 2)
    ijkn8 = z.reshape(-1, 8, 3)

    ijkn8[ijkn8 > (nb - 1)] = nb - 1
    return ijkn8


def ijk_to_idx(ijk, nb):
    idx = ijk[..., 0] * nb * nb + ijk[..., 1] * nb + ijk[..., 2]
    return idx


# def idx_to_ijk(idx, nb):
#     i = idx // (nb * nb)
#     j = (idx - i * nb * nb) // nb
#     k = idx - i * nb * nb - j * nb
#     return i, j, k


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
    ijkn8 = ijk_to_ijkn8(ijk, nb)
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

    args:
        tuple_mesh: the input mesh
        index: index use to split the mesh

    returns:
        meshes: the dict of meshes

    name convetion:
        ui = unique index
    '''
    meshes = {}
    v, f = tuple_mesh
    ui = np.unique(idx)
    ui = ui[ui >= 0]
    for group_idx in ui:
        a, b = np.where(idx == group_idx)
        group_mesh = clean_order_mesh(v, f[a])
        meshes[group_idx] = (group_mesh[0], np.array(group_mesh[1], np.int32))
    return meshes


def split_mesh(tm, nb):
    ''' split tuple mesh given split information

    args:
        tm: a tuple with
            [0]: numpy array of vertices
            [1]: numpy array of faces
        nb: number of blocks along each axis

    returns:
        meshes: a dict of tuple meshes

    name convention:
        nb = number of blocks
        tm = tuple mesh
        bel = box edge length
    '''
    fc = get_face_centers(tm)
    bel = 1. / nb
    idx = convert_fc_to_idx(fc, bel, nb)
    meshes = convert_idx_to_meshes(tm, idx)
    return meshes


def load_tuple_mesh(fn):
    mesh = trimesh.load_mesh(fn, process=False)
    return mesh.vertices, mesh.faces



def save_mesh(v, f, fn):
    ''' save the mesh as an obj file

    args:
        v: vertices
        f: faces
        fn: save file name

    name convention:
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


def process_flatten_meshes(
    vs, vms, fs, fms, gi, y, **kwargs
):

    pass

def entry():
    pass
