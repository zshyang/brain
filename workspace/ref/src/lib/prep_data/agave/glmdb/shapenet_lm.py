'''
Zhangsihao Yang
04/10/2022

shapenet large mesh lmdb file generator

interactive -p htc -N 1 -c 20 --mem=50G
module load singularity/3.8.0
singularity exec \
-B /scratch/:/scratch/ \
/scratch/zyang195/singularity/occo.simg \
bash
python /scratch/zyang195/projects/base/src/lib/prep_data/agave/glmdb/shapenet_lm.py

reference:
https://github.com/rmccorm4/PyTorch-LMDB/blob/master/folder2lmdb.py

dp = dataset path
sp = save path
'''
import argparse
import json
import os
import pickle
import random
import string
import sys
from glob import glob

import lmdb
import msgpack
import numpy as np
# This segfaults when imported before torch: https://github.com/apache/arrow/issues/2637
import pyarrow as pa
import six
import torch
import torch.utils.data as data
import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm


def read_obj(obj_path):
    """Read vertices and faces from .obj file.
    This function will reorder the vertices and faces. The principle of the 
    ordering is based on the the order of the faces. The first vertex index
    apeared in the first face will be 0.
    This function is copyied from:
    https://github.com/deepmind/deepmind-research/blob/master/polygen/data_utils.py
    I have made two changes. And I am not sure about the robustness of this 
    function. 

    Args:
        obj_path: The path of the obj file.

    Returns:
        The vertices of the mesh.
        The faces of the mesh.
    """
    vertex_list = []  # the list of the vertices
    flat_vertices_list = []
    flat_vertices_indices = {}  # to store the name and the actual index
    flat_polygons = []

    with open(obj_path) as obj_file:  # open the file
        for line in obj_file:  # iterate the obj file
            tokens = line.split()
            if not tokens:  
                # If tokens are empty, then move to next line in the file.
                continue 
            line_type = tokens[0]
            # Skip lines not starting with v or f.
            if line_type == 'v':
                vertex_list.append(
                    [float(x) for x in tokens[1:]]
                )
            elif line_type == 'f':
                polygon = []
                for i in range(len(tokens) - 1):
                    # get the name of the vertex
                    vertex_name = tokens[i + 1].split(
                        '/'
                    )[0]
                    # The name of the vertex has been recorded before.
                    if vertex_name in flat_vertices_indices:
                        polygon.append(flat_vertices_indices[vertex_name])
                        continue
                    # The name of the vertex has not been recorded before.
                    flat_vertex = []
                    for index in six.ensure_str(vertex_name).split('/'):
                        # If the index is empty, then move to the next index in
                        # the vertex name.
                        if not index:
                            continue
                        # obj polygon indices are 1 indexed, so subtract 1 
                        # here.
                        flat_vertex += vertex_list[int(index) - 1]
                        # If it is "//", then only the first index is 
                        # meaningful. 
                        break
                    flat_vertices_list.append(flat_vertex)
                    # This is the change I have made. Because the face is start
                    # at 0. So here is a -1.
                    flat_vertex_index = len(flat_vertices_list) - 1
                    flat_vertices_indices[vertex_name] = flat_vertex_index
                    polygon.append(flat_vertex_index)
                flat_polygons.append(polygon)

    return np.array(flat_vertices_list, dtype=np.float32), flat_polygons


class MeshDataset(data.Dataset):
    def __init__(self, list_path):
        self.list_path = list_path

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, index):
        path = self.list_path[index]
        v, f = read_obj(path)
        try:
            f = np.array(f, dtype=np.float32)
            return path, v, f
        except:
            return [], [], []


class ImageFolderLMDB(data.Dataset):
    def __init__(
        self, db_path, transform=None, 
        target_transform=None
    ):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def folder2lmdb(list_path, outpath, write_frequency=5000):
    # directory = os.path.expanduser(path)
    # print("Loading dataset from %s" % directory)
    # dataset = ImageFolder(directory, loader=raw_reader)
    dataset = MeshDataset(list_path)

    data_loader = DataLoader(
        dataset, num_workers=20, collate_fn=lambda x: x,
    )

    # print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(
        outpath, subdir=False,
        map_size=1099511627776 * 2, readonly=False,
        meminit=False, map_async=True
    )

    txn = db.begin(write=True)
    counter = 0
    for idx, data in tqdm(enumerate(data_loader)):
        path, v, f = data[0]
        if path:
            txn.put(
                u'{}'.format(idx).encode('ascii'), 
                dumps_pyarrow((path, v, f))
            )
            counter += 1
        if idx % write_frequency == 0:
            print(f'[{idx}/{len(data_loader)}]')
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [
        u'{}'.format(k).encode('ascii') for k in range(
            counter + 1
        )
    ]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


def generate_json(dp):
    path_pattern = os.path.join(
        dp, '*', '*', 'models', '*.obj'
    )
    list_path = glob(path_pattern)
    json_path = os.path.join(dp, 'name.json')

    # make sure the file exists and is long
    if os.path.exists(json_path):
        with open(json_path, 'r') as jf:
            lp = json.load(jf)
            if len(lp) > 50000:
                return

    with open(json_path, 'w') as jf:
        json.dump(list_path, jf)


def load_json(dp):
    json_path = os.path.join(dp, 'name.json')

    # make sure the file exists and is long
    if os.path.exists(json_path):
        with open(json_path, 'r') as jf:
            lp = json.load(jf)
    return lp


def split_train_val(list_path):
    random.seed(0)
    random.shuffle(list_path)
    p = int(len(list_path) * 0.85)
    list_train = list_path[:p]
    list_val = list_path[p:]
    return list_train, list_val


def main():
    dp = '/scratch/zyang195/dataset/shapenet/mansim/'
    generate_json(dp)
    list_path = load_json(dp)
    list_train, list_val = split_train_val(list_path)

    sp = (
        '/scratch/zyang195/dataset/shapenet/mansim/'
        'lmdb/train.lmdb'
    )
    folder2lmdb(list_train, sp, write_frequency=1000)
    sp = (
        '/scratch/zyang195/dataset/shapenet/mansim/'
        'lmdb/val.lmdb'
    )
    folder2lmdb(list_val, sp, write_frequency=1000)


if __name__ == '__main__':
    main()
