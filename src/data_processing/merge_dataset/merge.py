''' merge dataset 

author:
    Zhangsihao Yang (zshyang1106@gmail.com)

logs:
    2023-02-13: init
'''
import os
from glob import glob

from mfile_loader import MFileLoader


def parse_all_path(file_path):
    '''parse the file path
    '''
    loaded_mfile = MFileLoader(file_path)
    print(loaded_mfile)
    split_file_path = file_path.split('/')
    print(file_path)
    dataset_type = 'OASIS'
    side_info = split_file_path[-2]
    stage = ''
    ori_id = split_file_path[-1].split('.')[0]
    print(ori_id)

    file_name = file_path[-1]
    file_name = file_name.split('.')[0]
    file_name = file_name.split('_')
    file_name = file_name[0] + '_' + file_name[1] + '_' + file_name[2]
    file_name = file_name + '.obj'
    file_path = '/'.join(file_path[:-1])
    file_path = file_path + '/' + file_name
    return file_path


def main():
    save_folder_path = '/workspace/data/merged_dataset/raw'
    all_file_paths = glob('/workspace/data/all/*/*.m')
    for all_file_path in all_file_paths:
        parse_all_path(all_file_path)
        break

    adni_folder_path = glob('/workspace/data/MMS/*/*/*.m')


if __name__ == '__main__':
    main()
