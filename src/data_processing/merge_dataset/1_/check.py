'''
author;
    Zhangsihao Yang

logs:
    2023-02-18: create
'''
from glob import glob
import os

def check():
    root_path = '/scratch/zyang195/projects/brain/data/merged/raw'
    print('the length of the jfeature', len(glob(root_path + '/*_jfeature.npy')))
    for i in range(2296):
        if not os.path.exists(root_path + f'/{i:04d}_simpilified_jfeature.npy'):
            print(i)

    print('the length of simplified mesh', len(glob(root_path + '/*_simplified.obj')))
    for i in range(2296):
        if not os.path.exists(root_path + f'/{i:04d}_simplified.obj'):
            print(i)


def get_file_index():
    # 707
    # 798
    # 1017
    # 1062
    # 1584
     # get the list of files
    file_paths = glob('/scratch/zyang195/projects/brain/data/merged/raw/*.npz')
    for i in range(len(file_paths)):
        if '1584' in file_paths[i]:
            print('found')
            print(i)
            break

if __name__ == '__main__':
    check()
    get_file_index()
