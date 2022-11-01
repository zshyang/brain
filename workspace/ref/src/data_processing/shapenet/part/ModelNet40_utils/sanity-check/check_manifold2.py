"""
Author: Vinit V. Singh
Checks for non-manifold decimated-mesh
"""
import pymesh
from file_utils import fpath

DECIMATE = '../ModelNet40-decimate-1024'
f_path_dcmt = fpath(DECIMATE)

count_nmfld = 0
for path in f_path_dcmt:
    mesh = pymesh.load_mesh(path)
    if not mesh.is_manifold():
        print('Non-manifold mesh at : {0}'.format(path))
        count_nmfld += 1

print('Number of non-manifold meshes: {0}'.format(count_nmfld))
