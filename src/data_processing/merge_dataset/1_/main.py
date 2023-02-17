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


def make_file_folder(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def write_obj_file(obj_file, verts, faces):
    make_file_folder(obj_file)
    with open(obj_file, "w") as file:
        for vert in verts:
            file.write("v {} {} {}\n".format(vert[0], vert[1], vert[2]))
        for face in faces:
            file.write("f {} {} {}\n".format(face[0], face[1], face[2]))


SIMPLIFY_PATH = 'manifold/simplify'
TARGET_FACE_NUM = 2048 * 4


def get_path(obj_file):
    split_path = obj_file.split('/')
    split_path[1] = 'simplified'
    return '/'.join(split_path)


def make_file_folder(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def generate_file(obj_file, simplified_mesh_path):
    make_file_folder(simplified_mesh_path)
    cmd = f'{SIMPLIFY_PATH} -i {obj_file} -o {simplified_mesh_path} -m -f {TARGET_FACE_NUM}'
    os.system(cmd)


def main():
    list_obj_file = glob('../manifold/*/l/*.obj')
    for obj_file in tqdm(list_obj_file):
        path = get_path(obj_file)
        generate_file(obj_file, path)


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
    jfeatures = 
    # load the remeshed mesh
    # compute the cloese face on the oringinal mesh for each vertex on the remeshed mesh
    # compute the Jfeature on the remeshed mesh using barycentric coordinates

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='merge dataset')

    parser.add_argument(
        '--index', type=int, default='0', help='the index in the list'
    )

    args = parser.parse_args()

    main(args)
