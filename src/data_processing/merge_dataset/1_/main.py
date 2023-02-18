'''
author:
    Zhangsihao Yang

logs:
    2023-02-15: init
'''

import argparse
import os
from glob import glob

import numpy as np
from scipy.spatial import cKDTree


def make_file_folder(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def write_obj_file(obj_file, verts, faces):
    make_file_folder(obj_file)
    with open(obj_file, "w") as file:
        for vert in verts:
            file.write("v {} {} {}\n".format(vert[0], vert[1], vert[2]))
        for face in faces:
            file.write("f {} {} {}\n".format(face[0], face[1], face[2]))


def parse_vertex(line, verts):
    if line[0] == 'v':
        verts.append([float(x) for x in line.split()[1:]])


def parse_face(line, faces):
    if line[0] == 'f':
        faces.append([int(x.split('/')[0]) for x in line.split()[1:]])


def load_obj_file(obj_file):
    verts = []
    faces = []
    with open(obj_file, "r") as file:
        for line in file:
            parse_vertex(line, verts)
            parse_face(line, faces)
    return np.array(verts), np.array(faces)


def compute_closest_faces(orig_vertices, remeshed_vertices, orig_faces):
    ''' compute the closest faces for each remeshed vertex

    Args:
        orig_vertices (np.ndarray): the vertices of the original mesh
        remeshed_vertices (np.ndarray): the vertices of the remeshed mesh
        orig_faces (np.ndarray): the faces of the original mesh
    '''
    face_centers = orig_vertices[orig_faces - 1].mean(-2)

    orig_kdtree = cKDTree(face_centers)

    closest_faces = np.zeros(len(remeshed_vertices), dtype=int)
    for i, vertex in enumerate(remeshed_vertices):
        _, orig_vertex_idx = orig_kdtree.query(vertex)
        closest_faces[i] = orig_vertex_idx

    return closest_faces


def project_point_onto_plane(remeshed_verts, vertices, faces, closest_faces):
    faces = faces - 1
    # compute the normal of each face
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                            vertices[faces[:, 2]] - vertices[faces[:, 0]])
    face_normals = face_normals / \
        np.linalg.norm(face_normals, axis=1, keepdims=True)

    # get the normal for the closest face
    selected_face_normals = face_normals[closest_faces]
    # get the first point on the selected face
    selected_face_q_vertices = vertices[faces[closest_faces][:, 0]]
    # compute the vector from the remeshed vertices to the selected face
    remeshed_to_q = remeshed_verts - selected_face_q_vertices
    # compute the distance of the remeshed vertices to the selected face
    dist = np.sum(remeshed_to_q * selected_face_normals, axis=1)
    # project the remeshed vertices onto the selected face
    projected_verts = remeshed_verts - \
        dist[:, np.newaxis] * selected_face_normals
    return projected_verts


def compute_barycentric_coords(projected_verts, closest_faces, vertices, faces):
    faces = faces - 1

    A = vertices[faces[closest_faces][:, 0]]
    B = vertices[faces[closest_faces][:, 1]]
    C = vertices[faces[closest_faces][:, 2]]

    P = projected_verts

    v1 = B - A
    v2 = C - A
    v3 = P - A

    dot11 = (v1 * v1).sum(axis=1)
    dot12 = (v1 * v2).sum(axis=1)
    dot22 = (v2 * v2).sum(axis=1)
    dot31 = (v3 * v1).sum(axis=1)
    dot32 = (v3 * v2).sum(axis=1)

    invDenom = 1 / (dot11 * dot22 - dot12 * dot12)
    u = (dot22 * dot31 - dot12 * dot32) * invDenom
    v = (dot11 * dot32 - dot12 * dot31) * invDenom
    w = 1.0 - u - v

    return np.array([u, v, w]).T


def main(args):
    index = args.index

    # get the list of files
    file_paths = glob('/workspace/data/merged/raw/*.npz')
    face_path = '/workspace/data/merged/face.npy'

    # load the npz file
    npz_data = np.load(file_paths[index])
    faces = np.load(face_path)

    # ==== make the mesh water tight ==== #
    # load the mesh
    vertices = npz_data['vertices']
    # save the mesh as obj
    file_index = file_paths[index].split('/')[-1].split('.')[0]
    obj_file = f'/workspace/data/merged/raw/{file_index}.obj'
    write_obj_file(obj_file, vertices, faces)
    # manifold the mesh
    WT_PATH = '/workspace/src/data_processing/manifold/manifold'
    manifold_path = f'/workspace/data/merged/raw/{file_index}_manifold.obj'
    cmd = f'{WT_PATH} {obj_file} {manifold_path}'
    os.system(cmd)
    # remove the obj file
    os.remove(obj_file)

    # ==== simplify the mesh ==== #
    SIMPLIFY_PATH = '/workspace/src/data_processing/manifold/simplify'
    target_face_num = 2048 * 4
    simplified_mesh_path = f'/workspace/data/merged/raw/{file_index}_simplified.obj'
    # simplify the mesh
    cmd = f'{SIMPLIFY_PATH} -i {manifold_path} -o {simplified_mesh_path} -m -f {target_face_num}'
    os.system(cmd)
    # remove the water tight mesh
    os.remove(manifold_path)

    # ==== resample the Jfeature ==== #
    # load the Jfeature
    jfeatures = npz_data['jfeature']
    # load the remeshed mesh
    remeshed_verts, remeshed_faces = load_obj_file(simplified_mesh_path)
    # compute the cloese face on the oringinal mesh for each vertex on the remeshed mesh
    closest_faces = compute_closest_faces(
        vertices, remeshed_verts, faces)  # (Nof,)
    # project remeshed vertices to the closest face on the original mesh
    projected_verts = project_point_onto_plane(
        remeshed_verts, vertices, faces, closest_faces)
    # compute the barycentric coordinates for each vertex on the remeshed mesh
    barycentric_coords = compute_barycentric_coords(
        projected_verts, closest_faces, vertices, faces)
    # compute the Jfeature on the remeshed mesh using barycentric coordinates
    remeshed_jfeatures = np.sum(np.expand_dims(
        barycentric_coords, axis=-1) * jfeatures[faces[closest_faces] - 1], axis=1)
    # save the remeshed Jfeature
    np.save(
        f'/workspace/data/merged/raw/{file_index}_simpilified_jfeature.npy', remeshed_jfeatures)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='merge dataset')

    parser.add_argument(
        '--index', type=int, default='0', help='the index in the list'
    )

    args = parser.parse_args()

    main(args)
