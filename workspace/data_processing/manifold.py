import os
from glob import glob

from tqdm import tqdm

WT_PATH = 'manifold/manifold'

def get_manifold_path(obj_file):
    split_path = obj_file.split('/')
    split_path[1] = 'manifold'
    return '/'.join(split_path)


def make_file_folder(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def generate_manifold_file(obj_file, manifold_path):
    make_file_folder(manifold_path)
    cmd = f'{WT_PATH} {obj_file} {manifold_path}'
    os.system(cmd)
    

def main():
    list_obj_file = glob('../obj/*/l/*.obj')
    for obj_file in tqdm(list_obj_file):
        manifold_path = get_manifold_path(obj_file)
        generate_manifold_file(obj_file, manifold_path)


if __name__ == '__main__':
    main()
