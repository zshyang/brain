"""
Author: Vinit V. Singh
Checks for non-manifold decimated-mesh
"""
import torch
from file_utils import fpath
from mesh_utils import pytorch3D_mesh, open3D_mesh, is_manifold

device = torch.device('cpu:0')
DECIMATE = '../ModelNet40-decimate-1024'
f_path_dcmt = fpath(DECIMATE)

count_nmfld = 0
for path in f_path_dcmt:
    mesh, faces, verts, verts_normals, edges = pytorch3D_mesh(path, device)
    mesh, faces, verts, verts_normals, edges = open3D_mesh(faces, verts, verts_normals, edges)
    if not is_manifold(mesh):
        print('Non-manifold mesh at : {0}'.format(path))
        count_nmfld += 1

print('Number of non-manifold meshes: {0}'.format(count_nmfld))
