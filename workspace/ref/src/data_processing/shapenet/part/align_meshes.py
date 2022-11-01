''' followed by simplify_meshes.py 

author:
    zhangsihao yang

logs:
    20220917
        file created
'''
import json
import os

import numpy as np
import trimesh
from tqdm import tqdm

SOURCE_DIR = '/datasets/shapenet/part/mesh/simplified'
TARGET_DIR = '/datasets/shapenet/part/mesh/aligned'


def remove_file(file_path):
    os.system(f'rm {file_path}')


def get_simplified_mesh_path(file_path):
    split_file_path = file_path.split('/')
    return os.path.join(SOURCE_DIR, split_file_path[1], split_file_path[2], 'simplified.obj')


def get_aligned_mesh_path(file_path):
    split_file_path = file_path.split('/')
    return os.path.join(TARGET_DIR, split_file_path[1], split_file_path[2], 'aligned.obj')


def load_mesh(mesh_path):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    return vertices, faces


def get_point_path(file_path):
    root = '/datasets/shapenet/part/shapenetcore_partanno_segmentation_benchmark_v0_normal'
    split_file_path = file_path.split('/')
    
    point_path = os.path.join(root, f'{split_file_path[1]}/{split_file_path[2]}.txt')
    return point_path


def load_point(point_path):
    point = []
    with open(point_path, 'r') as point_file:
        for line in point_file.readlines():
            point.append(np.fromstring(line, sep=' ')[:3])
    point = np.stack(point)
    return point


def align_mesh_with_point(tuple_mesh, point):
    # flip tuple mesh
    vertices, faces = tuple_mesh
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    vertices = np.stack((-z, y, x), axis=1)

    # find the centers of tuple mesh and point
    center_point = (point.min(0) + point.max(0)) / 2.0
    center_vertices = (vertices.min(0) + vertices.max(0)) / 2.0

    vertices = vertices - center_vertices + center_point

    return vertices, faces


def save_mesh(aligned_mesh_path, aligned_mesh):
    vertices, faces = aligned_mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    save_string = trimesh.exchange.obj.export_obj(mesh)

    with open(aligned_mesh_path, 'w') as save_file:
        save_file.write(save_string)


def generate_aligned_mesh(simplified_mesh_path, aligned_mesh_path, file_path):
    tuple_mesh = load_mesh(simplified_mesh_path)

    point_path = get_point_path(file_path)
    point = load_point(point_path)

    aligned_mesh = align_mesh_with_point(tuple_mesh, point)

    os.makedirs(os.path.dirname(aligned_mesh_path), exist_ok=True)
    save_mesh(aligned_mesh_path, aligned_mesh)


def process_list(list_file_path):
    for file_path in tqdm(list_file_path):
        simplified_mesh_path = get_simplified_mesh_path(file_path)
        aligned_mesh_path = get_aligned_mesh_path(file_path)

        if not os.path.exists(aligned_mesh_path):
            # print(file_path)
            generate_aligned_mesh(simplified_mesh_path, aligned_mesh_path, file_path)
            # remove_file(simplified_mesh_path)


def load_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def get_list_file_path(file_path):
    file_path = os.path.join(SOURCE_DIR, file_path)
    return load_json(file_path)


def main():
    list_train_file_path = get_list_file_path('train.json')
    process_list(list_train_file_path)

    list_train_file_path = get_list_file_path('val.json')
    process_list(list_train_file_path)

    list_train_file_path = get_list_file_path('test.json')
    process_list(list_train_file_path)


if __name__ == '__main__':
    main()
