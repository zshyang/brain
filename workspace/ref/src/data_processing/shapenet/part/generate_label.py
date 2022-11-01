''' generate_label for the npz files in shapenet part.

author:
    zhangsihao yang

logs:
    20220917
        file created
    skip /datasets/shapenet/part/shapenetcore_partanno_segmentation_benchmark_v0/04379243/points/f09ef9a34df9b34d9420b255bb5956f0.pts
'''
import json
import os
from glob import glob

import numpy as np
import trimesh
from tqdm import tqdm

from main import compute_face_to_point_index


def load_mesh(mesh_path):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    return vertices, faces


def get_file_pattern(obj_file_path):
    split_file_path = obj_file_path.split('/')
    return [split_file_path[-3], split_file_path[-2]]


def get_point_label_path(file_pattern):
    point_label_root = '/datasets/shapenet/part/shapenetcore_partanno_segmentation_benchmark_v0'
    point_label_path = os.path.join(point_label_root, file_pattern[0], 'points_label', f'{file_pattern[1]}.seg')
    return point_label_path

def get_point_path(file_pattern):
    point_root = '/datasets/shapenet/part/shapenetcore_partanno_segmentation_benchmark_v0'
    point_path = os.path.join(point_root, file_pattern[0], 'points', f'{file_pattern[1]}.pts')
    return point_path


def gather_list_npz():
    file_pattern = os.path.join('/datasets/shapenet/part/mesh/aligned', '*/*', '*.npz')
    return glob(file_pattern)


def load_point(point_path):
    point = []
    with open(point_path, 'r') as point_file:
        for line in point_file.readlines():
            point.append(np.fromstring(line, sep=' '))
    point = np.stack(point)
    return point


def load_point_label(point_label_path):
    point_label = []
    with open(point_label_path, 'r') as point_label_file:
        for line in point_label_file.readlines():
            point_label.append(np.fromstring(line, sep=' '))
    point_label = np.stack(point_label)
    return point_label


def compute_point_to_face_index(point, aligned_mesh):
    vertices, faces = aligned_mesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    _, _, face_idx = trimesh.proximity.closest_point(mesh, point)

    return np.array(face_idx, np.int32)


def get_label_path(file_pattern):
    label_root = '/datasets/shapenet/part/mesh/label'
    return os.path.join(label_root, file_pattern[0], file_pattern[1], 'label.npz')


def save_label(
        label_path, face_to_point_index, point_to_face_index,
        point_label, face_label
    ):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    # with open(label_path, 'w') as label_file:
    #     json.dump(
    #         {
    #             'face_to_point_index': face_to_point_index, 
    #             'point_to_face_index': point_to_face_index,
    #             'point_label': point_label,
    #             'face_label': face_label
    #         },
    #         label_file
    #     )
    np.savez(
        label_path,
        face_to_point_index=face_to_point_index,
        point_to_face_index=point_to_face_index,
        point_label=point_label,
        face_label=face_label
    )


def process_list_npz(list_npz_file):
    for npz_file in tqdm(list_npz_file):
        obj_file = npz_file.replace('.npz', '.obj')
        file_pattern = get_file_pattern(obj_file)
        label_path = get_label_path(file_pattern)

        if os.path.exists(label_path):
            continue

        tuple_mesh = load_mesh(obj_file)

        # mesh = trimesh.load_mesh(obj_file, process=False)
        # npz = np.load(npz_file)

        point_label_path = get_point_label_path(file_pattern)
        point_path = get_point_path(file_pattern)

        try:
            point = load_point(point_path)
        except:
            print(f'skip {point_path}')
            continue

        point_label = load_point_label(point_label_path)

        face_to_point_index = compute_face_to_point_index(tuple_mesh, point)
        point_to_face_index = compute_point_to_face_index(point, tuple_mesh)
        # point_to_face_index = project_point_to_mesh(point, aligned_mesh)

        face_label = point_label[face_to_point_index]


        save_label(
            label_path, face_to_point_index, point_to_face_index,
            point_label, face_label
        )

        # print(label_path)
        # break


def main():
    list_npz_file = gather_list_npz()
    process_list_npz(list_npz_file)


if __name__ == '__main__':
    main()
