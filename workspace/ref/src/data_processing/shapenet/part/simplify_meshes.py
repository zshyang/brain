''' make the mesh watertight, simplify the mesh and align the mesh with the point cloud.

author:
    zhangsihao yang

logs:
    20220917
        file created
'''
import json
import os

from tqdm import tqdm

SOURCE_DIR = '/home/george/George/datasets/shapenet/part/mesh/part'
TARGET_DIR = '/home/george/George/datasets/shapenet/part/mesh/simplified'
WT_PATH = '/home/george/George/projects/manifold/Manifold/build/manifold'
SIMPLIFY_PATH = '/home/george/George/projects/manifold/Manifold/build/simplify'

def load_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def get_ori_mesh_path(file_path):
    split_file_path = file_path.split('/')
    return os.path.join(SOURCE_DIR, split_file_path[1], split_file_path[2], split_file_path[2], 'models', 'model_normalized.obj')


def get_wt_mesh_path(file_path):
    split_file_path = file_path.split('/')
    return os.path.join(TARGET_DIR, split_file_path[1], split_file_path[2], 'wt.obj')


def get_simplified_mesh_path(file_path):
    split_file_path = file_path.split('/')
    return os.path.join(TARGET_DIR, split_file_path[1], split_file_path[2], 'simplified.obj')


def get_aligned_mesh_path(file_path):
    split_file_path = file_path.split('/')
    return os.path.join(TARGET_DIR, split_file_path[1], split_file_path[2], 'aligned.obj')


def remove_file(file_path):
    os.system(f'rm {file_path}')


def generate_wt_mesh(ori_mesh_path, wt_mesh_path):
    os.makedirs(os.path.dirname(wt_mesh_path), exist_ok=True)
    cmd = f'{WT_PATH} {ori_mesh_path} {wt_mesh_path}'
    os.system(cmd)


def generate_simplified_mesh(wt_mesh_path, simplified_mesh_path):
    os.makedirs(os.path.dirname(simplified_mesh_path), exist_ok=True)
    cmd = f'{SIMPLIFY_PATH} -i {wt_mesh_path} -o {simplified_mesh_path} -m -f 1024'
    os.system(cmd)


def generate_aligned_mesh(simplified_mesh_path, aligned_mesh_path):
    os.makedirs(os.path.dirname(wt_mesh_path), exist_ok=True)
    cmd = f'{WT_PATH} {ori_mesh_path} {wt_mesh_path}'
    os.system(cmd)


def process_list(list_file_path):
    for file_path in tqdm(list_file_path):
        ori_mesh_path = get_ori_mesh_path(file_path)
        wt_mesh_path = get_wt_mesh_path(file_path)
        simplified_mesh_path = get_simplified_mesh_path(file_path)
        # aligned_mesh_path = get_aligned_mesh_path(file_path)

        generate_wt_mesh(ori_mesh_path, wt_mesh_path)
        generate_simplified_mesh(wt_mesh_path, simplified_mesh_path)
        # generate_aligned_mesh(simplified_mesh_path, aligned_mesh_path)

        remove_file(wt_mesh_path)
        # remove_file(simplified_mesh_path)

        # break


def process_train():
    # load train json file
    list_path = load_json(os.path.join(SOURCE_DIR, 'train_test_split', 'shuffled_train_file_list.json'))

    # pich first 5%
    end_index = int(len(list_path) * 0.05)
    list_path = list_path[:end_index]

    process_list(list_path)


def process_val():
    # load val json file
    list_path = load_json(os.path.join(SOURCE_DIR, 'train_test_split', 'shuffled_val_file_list.json'))
    process_list(list_path)


def process_test():
    # load test json file
    list_path = load_json(os.path.join(SOURCE_DIR, 'train_test_split', 'shuffled_test_file_list.json'))
    process_list(list_path)


def main():
    process_train()
    process_val()
    process_test()


def get_num():
    list_path = load_json(os.path.join(SOURCE_DIR, 'train_test_split', 'shuffled_train_file_list.json'))

    # pich first 5%
    end_index = int(len(list_path) * 0.05)
    list_path = list_path[:end_index]
    print(len(list_path))

    # load val json file
    list_path = load_json(os.path.join(SOURCE_DIR, 'train_test_split', 'shuffled_val_file_list.json'))
    print(len(list_path))

    # load test json file
    list_path = load_json(os.path.join(SOURCE_DIR, 'train_test_split', 'shuffled_test_file_list.json'))
    print(len(list_path))


if __name__ == '__main__':
    # main()
    get_num()
