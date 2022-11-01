"""
Author: Vinit V. Singh
Checks for if number of faces are equal to the desired face count in the decimated-mesh
"""
import os
import torch
import open3d as o3d
from file_utils import fpath
from mesh_utils import pytorch3D_mesh

device = torch.device('cpu:0')
max_faces = 1024

DECIMATE = '../ModelNet40-decimate-1024'
f_path_dcmt = fpath(DECIMATE)

unequal_face = 0
for path in f_path_dcmt:
    mesh, faces, verts, verts_normals, edges = pytorch3D_mesh(path, device)
    if faces.shape[0] != max_faces:
        print('Mesh with faces not equal to {0} faces at {1}'.format(max_faces, path))
        unequal_face += 1

print('Number of mesh with faces not equal to {0} faces: {1}'.format(max_faces, unequal_face))
