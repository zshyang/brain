''' sample points from meshes

author:
    zhangsihao yang

logs:
    20221030    create file
'''
import os
from glob import glob

import numpy as np
import trimesh
from tqdm import tqdm

NUM_SAMPLE = 1024 * 16

def get_sample_point_path(file_path):
    split_file_path = file_path.split('/')
    split_file_path[1] = 'sample_point'
    file_path = '/'.join(split_file_path)
    split_file_path = file_path.split('.')
    split_file_path[-1] = 'pts'
    return '.'.join(split_file_path)


def sample(file_path):
    mesh = trimesh.load_mesh(file_path)
    points, _ = trimesh.sample.sample_surface(mesh, count=NUM_SAMPLE)
    return np.array(points, dtype=np.float32)


def make_file_folder(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def save_sample_points(sample_points, sample_point_path):
    make_file_folder(sample_point_path)
    with open(sample_point_path, 'w') as open_file:
        for point in sample_points:
            open_file.write(f'{point[0]} {point[1]} {point[2]}\n')


def main():
    list_file = glob('../manifold/*/l/*.obj')
    for file_path in tqdm(list_file):
        sample_point_path = get_sample_point_path(file_path)
        sample_points = sample(file_path)
        save_sample_points(sample_points, sample_point_path)


if __name__ == '__main__':
    main()
