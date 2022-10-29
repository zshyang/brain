import os
from glob import glob

from tqdm import tqdm

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


if __name__ == '__main__':
    main()
