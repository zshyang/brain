'''
author:
    zhangsihao yang

logs:
    20220916
        file created
'''
import json
import os

from tqdm import tqdm

SOURCE_DIR = '/scratch/zyang195/datasets/shapenet/ShapeNetCore.v2/'
TARGET_DIR = '/scratch/zyang195/datasets/shapenet/part/'

def load_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def find_source_folder_path(file_path):
    split_file_path = file_path.split('/')
    source_folder_path = os.path.join(SOURCE_DIR, split_file_path[1], split_file_path[2])
    return source_folder_path + '/'


def get_target_folder_path(file_path):
    split_file_path = file_path.split('/')
    target_folder_path = os.path.join(TARGET_DIR, split_file_path[1], split_file_path[2])
    return target_folder_path + '/'


def copy_folder(source_folder_path, target_folder_path):
    cmd = f'mkdir -p {target_folder_path}'
    os.system(cmd)

    cmd = f'cp -R {source_folder_path} {target_folder_path}'
    os.system(cmd)


def process_list(list_file_path):
    for file_path in tqdm(list_file_path):
        source_folder_path = find_source_folder_path(file_path)
        target_folder_path = get_target_folder_path(file_path)
        copy_folder(source_folder_path, target_folder_path)


def process_train():
    # load train json file
    list_path = load_json(os.path.join(TARGET_DIR, 'train_test_split', 'shuffled_train_file_list.json'))

    # pich first 5%
    end_index = int(len(list_path) * 0.05)
    list_path = list_path[:end_index]

    process_list(list_path)


def process_val():
    # load val json file
    list_path = load_json(os.path.join(TARGET_DIR, 'train_test_split', 'shuffled_val_file_list.json'))
    process_list(list_path)


def process_test():
    # load test json file
    list_path = load_json(os.path.join(TARGET_DIR, 'train_test_split', 'shuffled_test_file_list.json'))
    process_list(list_path)


def main():
    process_train()
    process_val()
    process_test()


if __name__ == '__main__':
    main()
