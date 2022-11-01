''' generate list of json file

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

def load_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def get_simplified_mesh_path(file_path):
    split_file_path = file_path.split('/')
    return os.path.join(TARGET_DIR, split_file_path[1], split_file_path[2], 'simplified.obj')


def generate_json(json_file_name, list_path):
    new_list_path = []

    for file_path in tqdm(list_path):
        simplified_mesh_path = get_simplified_mesh_path(file_path)
        if os.path.exists(simplified_mesh_path):
            new_list_path.append(file_path)

    with open(os.path.join(TARGET_DIR, json_file_name), 'w') as path_file:
        json.dump(new_list_path, path_file)


def main():
    list_path = load_json(os.path.join(SOURCE_DIR, 'train_test_split', 'shuffled_train_file_list.json'))

    # pick first 5%
    end_index = int(len(list_path) * 0.05)
    list_path = list_path[:end_index]

    generate_json('train.json', list_path)

    list_path = load_json(os.path.join(SOURCE_DIR, 'train_test_split', 'shuffled_val_file_list.json'))
    generate_json('val.json', list_path)

    list_path = load_json(os.path.join(SOURCE_DIR, 'train_test_split', 'shuffled_test_file_list.json'))
    generate_json('test.json', list_path)


if __name__ == '__main__':
    main()
