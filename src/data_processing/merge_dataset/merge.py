''' merge dataset 

author:
    Zhangsihao Yang (zshyang1106@gmail.com)

logs:
    2023-02-13: init
'''
import os
from glob import glob
import argparse
from mfile_loader import MFileLoader


def parse_all_path(file_path):
    '''parse the file path
    '''
    loaded_mfile = MFileLoader(file_path)

    # split the file path
    split_file_path = file_path.split('/')

    dataset_type = 'OASIS'
    side_info = split_file_path[-2]
    stage = ''
    ori_id = split_file_path[-1].split('.')[0]

    return loaded_mfile, dataset_type, side_info, stage, ori_id


def parse_mms_path(file_path):
    pass


def save_results(results, save_folder_path, index):
    pass


def process(args):
    save_folder_path = '/workspace/data/merged/raw'

    # create the list of files
    file_paths = glob('/workspace/data/all/*/*.m') + glob('/workspace/data/MMS/*/*/*.m')

    

    file_path = file_paths[args.index]

    if file_path.split('/')[3] == 'all':
        results = parse_all_path(file_path)
        save_results(results, save_folder_path, args.index)
    elif file_path.split('/')[3] == 'MMS':
        parse_mms_path(file_path)
        save_results(results, save_folder_path, args.index)
    else:
        print('error')

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='merge dataset')

    parser.add_argument(
        '--index', type=int, default='0', help='the index in the list'
    )

    args = parser.parse_args()

    process(args)
