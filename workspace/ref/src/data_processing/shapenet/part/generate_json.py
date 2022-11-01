''' after run exps/shapenetpart/meshnet2/ssl_mae_test/r.sh
features are generated but train.json, val.json and test.json are
needed in order to train the mlp.

author:
    zhangsihao yang

logs:
    20220918
        file created
'''
import json
import os
from glob import glob

from tqdm import tqdm

SIMPLIFIED_PATH = '/home/george/George/datasets/shapenet/part/mesh/simplified'
FEATURE_PATH = '/home/george/George/datasets/shapenet/part/mesh/feature'

def load_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def get_file_path(feature_file_path):
    split_string = feature_file_path.split('/')
    return '/'.join(['shape_data', split_string[-3], split_string[-2]])


def get_list_feature_file_path():
    list_path = glob(os.path.join(FEATURE_PATH, '*/*/*.npy'))
    return list_path


def main():
    list_feature_file_path = get_list_feature_file_path()
    list_file_path = [
        get_file_path(feature_file_path) for feature_file_path in list_feature_file_path
    ]
    list_train_path = load_json(
        os.path.join(SIMPLIFIED_PATH, 'train.json')
    )
    list_test_path = load_json(
        os.path.join(SIMPLIFIED_PATH, 'test.json')
    )
    list_val_path = load_json(
        os.path.join(SIMPLIFIED_PATH, 'val.json')
    )
    
    # train
    ftrp = []
    for file_path in tqdm(list_train_path):
        if file_path in list_file_path:
            ftrp.append(file_path)
    print(len(ftrp))
    with open(os.path.join(FEATURE_PATH, 'train.json'), 'w') as json_file:
        json.dump(ftrp, json_file)
    
    # test
    ftsp = []
    for file_path in tqdm(list_test_path):
        if file_path in list_file_path:
            ftsp.append(file_path)
    print(len(ftsp))
    with open(os.path.join(FEATURE_PATH, 'test.json'), 'w') as json_file:
        json.dump(ftsp, json_file)

    # val
    fvap = []
    for file_path in tqdm(list_val_path):
        if file_path in list_file_path:
            fvap.append(file_path)
    print(len(fvap))
    with open(os.path.join(FEATURE_PATH, 'val.json'), 'w') as json_file:
        json.dump(fvap, json_file)


if __name__ == '__main__':
    main()
