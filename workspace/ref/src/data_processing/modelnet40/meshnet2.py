'''
author:
    zhangsihao yang

date:
    20220907

logs:
    20220907
        file    created
'''
import os
import os.path as osp
from glob import glob

import numpy as np
from tqdm import tqdm


def main():

    list_file_names = glob('/dataset/modelnet/meshnet2/ModelNet40/*/*/*.npz')
    print(len(list_file_names))

    for file_name in tqdm(list_file_names):
        mesh = np.load(file_name)
        faces = mesh['faces']
        verts = mesh['verts']
        ring_1 = mesh['ring_1']
        ring_2 = mesh['ring_2']
        ring_3 = mesh['ring_3']


if __name__ == '__main__':
    main()
