''' merge dataset 

author:
    Zhangsihao Yang (zshyang1106@gmail.com)

logs:
    2023-02-13: init
'''
import argparse
import json
import os
from glob import glob

import numpy as np
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
    ''' parse the file path
    '''
    loaded_mfile = MFileLoader(file_path)

    # split the file path
    split_file_path = file_path.split('/')

    dataset_type = 'ADNI'
    side_info = split_file_path[-2]
    stage = split_file_path[-3]
    ori_id = split_file_path[-1].split('.')[0]

    return loaded_mfile, dataset_type, side_info, stage, ori_id


def save_results(results, save_folder_path, index):
    loaded_mfile, dataset_type, side_info, stage, ori_id = results

    # save the mesh and jfeature into npz file
    vertices = loaded_mfile.get_vertices()
    faces = loaded_mfile.get_faces()
    jfeatures = loaded_mfile.get_jfeatures()

    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    jfeature = np.array(jfeatures, dtype=np.float32)

    npz_path = os.path.join(save_folder_path, f'{index:04d}.npz')
    if os.path.exists(npz_path) is False:
        np.savez(
            npz_path,
            vertices=vertices,
            faces=faces,
            jfeature=jfeature
        )

    # save the meta information into json file
    meta_info = {
        'dataset_type': dataset_type,
        'side_info': side_info,
        'stage': stage,
        'ori_id': ori_id
    }
    meta_path = os.path.join(save_folder_path, f'{index:04d}.json')
    if os.path.exists(meta_path) is False:
        with open(meta_path, 'w') as f:
            json.dump(meta_info, f)


def process(args):
    save_folder_path = '/workspace/data/merged/raw'
    if os.path.exists(save_folder_path) is False:
        os.makedirs(save_folder_path)

    # create the list of files
    file_paths = glob('/workspace/data/all/*/*.m') + \
        glob('/workspace/data/MMS/*/*/*.m')

    file_path = file_paths[args.index]

    if file_path.split('/')[3] == 'all':
        results = parse_all_path(file_path)
        save_results(results, save_folder_path, args.index)
    elif file_path.split('/')[3] == 'MMS':
        results = parse_mms_path(file_path)
        save_results(results, save_folder_path, args.index)
    else:
        print('error')
        return None


    # just save the face
    if not os.path.exists('/workspace/data/merged/face.npy'):
        face = results[0].get_faces()
        face = np.array(face, dtype=np.int32)
        np.save('/workspace/data/merged/face.npy', face)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='merge dataset')

    parser.add_argument(
        '--index', type=int, default='0', help='the index in the list'
    )

    args = parser.parse_args()

    process(args)
