''' this file is running in singularity container

author:
    zhangsihao yang

logs:
    20230221: file created
'''
import os
from glob import glob

import numpy as np
import torch

try:
    import trimesh as t_mesh
except:
    os.system('pip install trimesh')
    import trimesh as t_mesh

from pytorch3d.structures import Meshes
from tqdm import tqdm
from trimesh.graph import face_adjacency
from utils import fpath, is_mesh_valid

DATA_ROOT = '/workspace/data/merged/raw'
TARGET_ROOT = '/workspace/data/merged/processed'
DEVICE = torch.device('cpu:0')
MAX_FACES = 1024 * 8


def compute_global_scale():
    ''' Compute the global scale of the dataset
    '''
    # get the stored global scale
    cache_path = os.path.join(DATA_ROOT, '../global_scale.cache')
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            global_scale = float(f.read())
        return global_scale

    # compute the global scale
    global_scale = 0
    obj_file_paths = glob(os.path.join(DATA_ROOT, '*simplified.obj'))
    for obj_file_path in tqdm(obj_file_paths, desc='Compute global scale'):
        mesh = t_mesh.load(obj_file_path)
        verts = np.array(mesh.vertices)
        verts_mean = (verts.max(0) + verts.min(0)) / 2
        verts = verts - verts_mean
        # compute the radius of the mesh
        scale = np.sqrt((verts * verts).sum(1)).max()
        if scale > global_scale:
            global_scale = scale

    # save the global scale
    with open(cache_path, 'w') as f:
        f.write(str(global_scale))

    return global_scale


def normalize_mesh(verts, faces, scale=None):
    """
    Normalize and center input mesh to fit in a sphere of radius 1 centered at (0,0,0)

    Args:
        mesh: pytorch3D mesh

    Returns:
        mesh, faces, verts, edges, v_normals, f_normals: normalized pytorch3D mesh and other mesh
        information
    """
    verts = verts - (verts.max(0)[0] + verts.min(0)[0]) / 2
    if scale is None:
        scale = max(verts.abs().max(0)[0])
    verts = verts / scale
    mesh = Meshes(verts=[verts], faces=[faces])
    faces = mesh.faces_packed().squeeze(0)
    verts = mesh.verts_packed().squeeze(0)
    edges = mesh.edges_packed().squeeze(0)
    v_normals = mesh.verts_normals_packed().squeeze(0)
    f_normals = mesh.faces_normals_packed().squeeze(0)

    return mesh, faces, verts, edges, v_normals, f_normals


class MeshRing():
    def __init__(self, vertices, faces) -> None:
        self.vertices = vertices
        self.faces = faces
        self.triangle_mesh = t_mesh.Trimesh(
            vertices=vertices, faces=faces, process=False)

        self.cache = {}

    @property
    def faces_neighbor_1st_ring(self):
        if 'faces_neighbor_1st_ring' in self.cache:
            return self.cache['faces_neighbor_1st_ring']

        # Neighbor faces index along edges, Edges along neighbor_faces
        faces_adjacency, edges_adjacency = face_adjacency(
            faces=self.faces.permute(1, 0),
            mesh=self.triangle_mesh,
            return_edges=True)

        faces_neighbor_1st_ring = []
        edges_neighbor_1ring = []

        # For each face get 1-Ring neighborhood along its edges
        # For each face get edge between face and neighbor faces
        for face_idx in range(MAX_FACES):
            face_dim_0 = np.argwhere(faces_adjacency[:, 0] == face_idx)
            face_dim_1 = np.argwhere(faces_adjacency[:, 1] == face_idx)

            face_neighbor_dim_0 = faces_adjacency[:, 0][face_dim_1]
            face_neighbor_dim_1 = faces_adjacency[:, 1][face_dim_0]

            face_neighbor_1st_ring = np.concatenate([face_neighbor_dim_0,
                                                    face_neighbor_dim_1])

            # Edge between face and neighbor faces
            face_edge = np.concatenate(
                [face_dim_0, face_dim_1]).reshape(-1)
            edge_neighbor_1ring = edges_adjacency[face_edge]

            faces_neighbor_1st_ring.insert(
                face_idx, face_neighbor_1st_ring)
            edges_neighbor_1ring.insert(face_idx, edge_neighbor_1ring)

        faces_neighbor_1st_ring = np.asarray(
            faces_neighbor_1st_ring).squeeze(2)
        edges_neighbor_1ring = np.asarray(edges_neighbor_1ring)

        # Each face is connected to 3 other faces in the 1st Ring
        assert faces_neighbor_1st_ring.shape == (MAX_FACES, 3)
        # Each face has 1 edge between itself and neighbor faces
        # 2 in last dim since each edge is composed of 2 vertices
        assert edges_neighbor_1ring.shape == (MAX_FACES, 3, 2)

        # add to cache
        self.cache['faces_neighbor_1st_ring'] = faces_neighbor_1st_ring

        return self.cache['faces_neighbor_1st_ring']

    @property
    def faces_neighbor_2nd_ring(self):
        if 'faces_neighbor_2nd_ring' in self.cache:
            return self.cache['faces_neighbor_2nd_ring']

        faces_neighbor_0th_ring = np.arange(MAX_FACES)
        faces_neighbor_2ring = self.faces_neighbor_1st_ring[self.faces_neighbor_1st_ring]
        faces_neighbor_0ring = np.stack([faces_neighbor_0th_ring]*3, axis=1)
        faces_neighbor_0ring = np.stack([faces_neighbor_0ring]*3, axis=2)

        dilation_mask = faces_neighbor_2ring != faces_neighbor_0ring
        faces_neighbor_2nd_ring = faces_neighbor_2ring[dilation_mask]
        faces_neighbor_2nd_ring = faces_neighbor_2nd_ring.reshape(
            MAX_FACES, -1)

        # For each face there are 6 neighboring faces in its 2-Ring neighborhood
        assert faces_neighbor_2nd_ring.shape == (MAX_FACES, 6)

        # add to cache
        self.cache['faces_neighbor_2nd_ring'] = faces_neighbor_2nd_ring

        return self.cache['faces_neighbor_2nd_ring']

    @property
    def faces_neighbor_3rd_ring(self):
        if 'faces_neighbor_3rd_ring' in self.cache:
            return self.cache['faces_neighbor_3rd_ring']

        faces_neighbor_3ring = self.faces_neighbor_2nd_ring[self.faces_neighbor_1st_ring]
        faces_neighbor_3ring = faces_neighbor_3ring.reshape(MAX_FACES, -1)

        faces_neighbor_3rd_ring = []
        for face_idx in range(MAX_FACES):
            face_neighbor_3ring = faces_neighbor_3ring[face_idx]
            for neighbor in range(3):
                face_neighbor_1st_ring = self.faces_neighbor_1st_ring[face_idx, neighbor]
                dilation_mask = np.delete(
                    np.arange(face_neighbor_3ring.shape[0]),
                    np.where(face_neighbor_3ring == face_neighbor_1st_ring)[0][0:2])
                face_neighbor_3ring = face_neighbor_3ring[dilation_mask]
            faces_neighbor_3rd_ring.insert(face_idx, face_neighbor_3ring)

        # For each face there are 12 neighboring faces in its 3-Ring neighborhood
        faces_neighbor_3rd_ring = np.array(faces_neighbor_3rd_ring)
        assert faces_neighbor_3rd_ring.shape == (MAX_FACES, 12)

        # add to cache
        self.cache['faces_neighbor_3rd_ring'] = faces_neighbor_3rd_ring

        return self.cache['faces_neighbor_3rd_ring']


def compute_meshnet_feature(path, device, global_scale):
    # load mesh
    mesh = t_mesh.load(path)
    verts = torch.from_numpy(np.array(mesh.vertices))
    faces = torch.from_numpy(np.array(mesh.faces))
    mesh = Meshes(verts=[verts], faces=[faces])

    # check if the mesh is valid
    if not is_mesh_valid(mesh):
        return None

    # assert the shape of the mesh face is equal to max_faces
    if faces.shape[0] != (MAX_FACES):
        print(f'skip {path}')
        return None

    # Normalize Mesh
    mesh, faces, verts, edges, v_normals, f_normals = normalize_mesh(
        verts=verts, faces=faces, scale=global_scale)

    mesh_ring = MeshRing(vertices=verts, faces=faces)

    # 1st-Ring
    faces_neighbor_1st_ring = mesh_ring.faces_neighbor_1st_ring

    # 2nd-Ring
    faces_neighbor_2nd_ring = mesh_ring.faces_neighbor_2nd_ring

    # 3rd-Ring
    faces_neighbor_3rd_ring = mesh_ring.faces_neighbor_3rd_ring

    return verts, faces, faces_neighbor_1st_ring, faces_neighbor_2nd_ring, faces_neighbor_3rd_ring


def main():
    global_scale = compute_global_scale()

    obj_file_paths = glob(os.path.join(DATA_ROOT, '*simplified.obj'))

    for obj_file_path in tqdm(obj_file_paths, desc='compute meshnet feature'):
        # get obj file path index
        obj_file_path_index = obj_file_path.split('/')[-1][:4]

        target_file_path = os.path.join(
            TARGET_ROOT, f'{obj_file_path_index}.npz')
        if os.path.exists(target_file_path):
            continue

        results = compute_meshnet_feature(obj_file_path, DEVICE, global_scale)

        if results is None:
            continue

        np.savez(
            target_file_path,
            verts=results[0], faces=results[1],
            ring_1=results[2], ring_2=results[3], ring_3=results[4]
        )


if __name__ == '__main__':
    main()
