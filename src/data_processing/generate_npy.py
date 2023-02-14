''' generate npy files in place

author:
    zhangsihao yang

logs:
    20221030    file created
'''
import os

import numpy as np
import torch
import trimesh as t_mesh
from pytorch3d.structures import Meshes
# from torch_geometric.data import Data
# from torch_geometric.utils import to_trimesh
from tqdm import tqdm
from trimesh.graph import face_adjacency
from utils import fpath, is_mesh_valid, normalize_mesh, pytorch3D_mesh

device = torch.device('cpu:0')
# To process the dataset enter the path where they are stored
data_root = '../simplified'
max_faces = 1024 * 8

if not os.path.exists(data_root):
    raise Exception('Dataset not found at {0}'.format(data_root))

fpath_data = fpath(data_root)

# print(len(fpath_data))
# print(fpath_data.sdfs())

for path in tqdm(fpath_data):

    if os.path.exists(path.replace('.obj', '.npz')):
        continue

    mesh, faces, verts, edges, v_normals, f_normals = pytorch3D_mesh(path, device)
    if not is_mesh_valid(mesh):
        raise ValueError('Mesh is invalid!')
    # assert faces.shape[0] == (max_faces)
    if faces.shape[0] != (max_faces):
        print(f'skip {path}')
        continue

    # Normalize Mesh
    mesh, faces, verts, edges, v_normals, f_normals = normalize_mesh(verts=verts, faces=faces)

    ########################################################################### 1st-Ring ###########################################################################
    # data = Data(pos=verts, edge_index=edges.permute(1, 0), face=faces.permute(1, 0))
    # trimesh = to_trimesh(data)
    trimesh = t_mesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Neighbor faces index along edges, Edges along neighbor_faces
    faces_adjacency, edges_adjacency = face_adjacency(faces=faces.permute(1, 0),
                                                      mesh=trimesh,
                                                      return_edges=True)

    faces_neighbor_1st_ring = []
    edges_neighbor_1ring = []

    # For each face get 1-Ring neighborhood along its edges
    # For each face get edge between face and neighbor faces
    for face_idx in range(max_faces):
        face_dim_0 = np.argwhere(faces_adjacency[:, 0] == face_idx)
        face_dim_1 = np.argwhere(faces_adjacency[:, 1] == face_idx)

        face_neighbor_dim_0 = faces_adjacency[:, 0][face_dim_1]
        face_neighbor_dim_1 = faces_adjacency[:, 1][face_dim_0]

        face_neighbor_1st_ring = np.concatenate([face_neighbor_dim_0,
                                                 face_neighbor_dim_1])

        # Edge between face and neighbor faces
        face_edge = np.concatenate([face_dim_0, face_dim_1]).reshape(-1)
        edge_neighbor_1ring = edges_adjacency[face_edge]

        faces_neighbor_1st_ring.insert(face_idx, face_neighbor_1st_ring)
        edges_neighbor_1ring.insert(face_idx, edge_neighbor_1ring)

    faces_neighbor_1st_ring = np.asarray(faces_neighbor_1st_ring).squeeze(2)
    edges_neighbor_1ring = np.asarray(edges_neighbor_1ring)

    # Each face is connected to 3 other faces in the 1st Ring
    assert faces_neighbor_1st_ring.shape == (max_faces, 3)
    # Each face has 1 edge between itself and neighbor faces
    # 2 in last dim since each edge is composed of 2 vertices
    assert edges_neighbor_1ring.shape == (max_faces, 3, 2)

    ########################################################################### 2nd-Ring ###########################################################################
    faces_neighbor_0th_ring = np.arange(max_faces)
    faces_neighbor_2ring = faces_neighbor_1st_ring[faces_neighbor_1st_ring]
    faces_neighbor_0ring = np.stack([faces_neighbor_0th_ring]*3, axis=1)
    faces_neighbor_0ring = np.stack([faces_neighbor_0ring]*3, axis=2)

    dilation_mask = faces_neighbor_2ring != faces_neighbor_0ring
    faces_neighbor_2nd_ring = faces_neighbor_2ring[dilation_mask]
    faces_neighbor_2nd_ring = faces_neighbor_2nd_ring.reshape(max_faces, -1)

    # For each face there are 6 neighboring faces in its 2-Ring neighborhood
    assert faces_neighbor_2nd_ring.shape == (max_faces, 6)

    ########################################################################### 3rd-Ring ###########################################################################
    faces_neighbor_3ring = faces_neighbor_2nd_ring[faces_neighbor_1st_ring]
    faces_neighbor_3ring = faces_neighbor_3ring.reshape(max_faces, -1)

    faces_neighbor_3rd_ring = []
    for face_idx in range(max_faces):
        face_neighbor_3ring = faces_neighbor_3ring[face_idx]
        for neighbor in range(3):
            face_neighbor_1st_ring = faces_neighbor_1st_ring[face_idx, neighbor]
            dilation_mask = np.delete(
                np.arange(face_neighbor_3ring.shape[0]),
                np.where(face_neighbor_3ring == face_neighbor_1st_ring)[0][0:2])
            face_neighbor_3ring = face_neighbor_3ring[dilation_mask]
        faces_neighbor_3rd_ring.insert(face_idx, face_neighbor_3ring)
    # For each face there are 12 neighboring faces in its 3-Ring neighborhood
    faces_neighbor_3rd_ring = np.array(faces_neighbor_3rd_ring)
    assert faces_neighbor_3rd_ring.shape == (max_faces, 12)

    corners = verts[faces.long()]
    # Each face is connected to 3 other faces in the 1st Ring
    assert corners.shape == (max_faces, 3, 3)

    centers = torch.sum(corners, axis=1)/3
    assert centers.shape == (max_faces, 3)

    corners = corners.reshape(-1, 9)
    assert f_normals.shape == (max_faces, 3)

    faces_feature = np.concatenate([centers, corners, f_normals], axis=1)
    assert faces_feature.shape == (max_faces, 15)

    np.savez(path.replace('.obj', '.npz'),
             verts=verts,
             faces=faces,
             ring_1=faces_neighbor_1st_ring,
             ring_2=faces_neighbor_2nd_ring,
             ring_3=faces_neighbor_3rd_ring)
