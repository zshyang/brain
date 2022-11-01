"""
Author: Vinit V. Singh
Checks for duplicate faces and vertices in the decimated-mesh
"""

import os
from pymesh import load_mesh, save_mesh
from pymesh import remove_isolated_vertices, remove_duplicated_vertices, remove_degenerated_triangles, remove_duplicated_faces
from file_utils import fpath

def clean_mesh(mesh):
    #[1]
    mesh_clean, _ = remove_duplicated_faces(mesh)
    mesh_clean, _ = remove_duplicated_vertices(mesh_clean)
    mesh_clean, _ = remove_degenerated_triangles(mesh_clean)
    mesh_clean, _ = remove_isolated_vertices(mesh_clean)
    return mesh_clean

DECIMATE = '../ModelNet40-decimate-1024'
f_path_dcmt = fpath(DECIMATE)

for path in f_path_dcmt:
    mesh = load_mesh(path)
    mesh_cleaned = clean_mesh(mesh)
    #If number of vertices and faces is less after cleaning it implies that
    #there are duplicated faces and vertices present in the mesh
    if mesh.num_faces != mesh_cleaned.num_faces:
        print('Duplicates faces in mesh: ', path)
    if mesh.num_vertices != mesh_cleaned.num_vertices:
        print('Duplicates vertices in mesh: ', path)

#References:
#https://humansthatmake.com/meshlab/#:~:text=By%20using%20the%20%E2%80%9CRemove%20Isolated,well%20as%20the%20pink%20dots.
