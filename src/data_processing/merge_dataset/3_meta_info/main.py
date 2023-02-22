'''
author:
    zhangsihao yang

logs:
    20230221: file created
'''
import json
import os
import random
from glob import glob

DATA_ROOT = '/scratch/zyang195/projects/brain/data/merged/raw'


def main():
    store_info = {'pos': [], 'neg': []}

    # get the list of json file path
    json_file_paths = glob(os.path.join(DATA_ROOT, '*.json'))

    for json_file_path in json_file_paths:
        with open(json_file_path, 'r') as f:
            meta_info = json.load(f)
        if meta_info['dataset_type'] == 'ADNI' and meta_info['side_info'] == 'l':
            # get the index of the file
            file_index = json_file_path.split('/')[-1].split('.')[0]

            stage = meta_info['stage']

            if stage == 'AD':
                store_info['pos'].append([file_index, stage])
            elif 'pos' in stage:
                store_info['pos'].append([file_index, stage])
            elif 'neg' in stage:
                store_info['neg'].append([file_index, stage])

    print(f'pos: {len(store_info["pos"])}')
    print(f'neg: {len(store_info["neg"])}')

    # random split the data
    random.shuffle(store_info['pos'])
    random.shuffle(store_info['neg'])

    # get the train validation and test data
    train_pos = store_info['pos'][:int(len(store_info['pos']) * 0.6)]
    train_neg = store_info['neg'][:int(len(store_info['neg']) * 0.6)]
    val_pos = store_info['pos'][
        int(len(store_info['pos']) * 0.6):int(len(store_info['pos']) * 0.8)]
    val_neg = store_info['neg'][
        int(len(store_info['neg']) * 0.6):int(len(store_info['neg']) * 0.8)]
    test_pos = store_info['pos'][int(len(store_info['pos']) * 0.8):]
    test_neg = store_info['neg'][int(len(store_info['neg']) * 0.8):]

    # print info
    print(f'train pos: {len(train_pos)}')
    print(f'train neg: {len(train_neg)}')
    print(f'val pos: {len(val_pos)}')
    print(f'val neg: {len(val_neg)}')
    print(f'test pos: {len(test_pos)}')
    print(f'test neg: {len(test_neg)}')

    # save the meta info
    with open(os.path.join(DATA_ROOT, '../meta_pos_neg.json'), 'w') as f:
        json.dump({
            'train_pos': train_pos,
            'train_neg': train_neg,
            'val_pos': val_pos,
            'val_neg': val_neg,
            'test_pos': test_pos,
            'test_neg': test_neg
        }, f)


if __name__ == '__main__':
    main()
