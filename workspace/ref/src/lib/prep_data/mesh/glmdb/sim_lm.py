'''
Zhangsihao Yang
04/16/2022

agave run:
interactive -p htc -N 1 -c 20 --mem=50G
module load singularity/3.8.0

singularity exec \
-B /scratch/:/scratch/ \
/scratch/zyang195/singularity/occo_v2.simg \
bash

python /scratch/zyang195/projects/base/src/lib/prep_data/mesh/glmdb/sim_lm.py

name convention:
ds = dataset
lt = list train
lv = list validation
of = open file
op = output path
td = train dataset
vd = validation dataset
wf = write frequency
'''
import json
import os
import pickle
import random
from glob import glob

import lmdb
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm


def dumps(obj):
    return pickle.dumps(obj, protocol=-1)


def loads(buf):
    return pickle.loads(buf)


def dataset2lmdb(ds, op, wf=1000):
    # dataset = MeshDataset(list_path)

    data_loader = DataLoader(
        ds, num_workers=20, 
        collate_fn=lambda x: x,
    )

    # print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(
        op, subdir=False,
        map_size=1099511627776 * 2, 
        readonly=False,
        meminit=False, map_async=True
    )

    txn = db.begin(write=True)
    counter = 0
    for idx, data in tqdm(enumerate(data_loader)):
        txn.put(
            u'{}'.format(idx).encode('ascii'), 
            dumps(data[0])
        )
        counter += 1
        if idx % wf == 0:
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
        txn.put(b'__keys__', dumps(keys))
        txn.put(b'__len__', dumps(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


def generate_json(dp):
    path_pattern = os.path.join(
        dp, '*', '*', '*.json'
    )
    json_path = os.path.join(dp, 'name.json')

    # make sure the file exists and is long
    if os.path.exists(json_path):
        with open(json_path, 'r') as jf:
            lp = json.load(jf)
            if len(lp) > 50000:
                return

    list_path = glob(path_pattern)
    with open(json_path, 'w') as jf:
        json.dump(list_path, jf)


def load_json(dp):
    json_path = os.path.join(
        dp, 'name.json'
    )

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


class DataSet(data.Dataset):
    def __init__(self, list_path):
        self.lp = list_path

    def __getitem__(self, index):
        path = self.lp[index]
        with open(path, 'r') as of:
            data = json.load(of)
        return data
    
    def __len__(self):
        return len(self.lp)


def main():
    dp = (
        '/scratch/zyang195/dataset'
        '/shapenet/voxsim/'
    )
    generate_json(dp)
    # print(dp.sdfsf())
    list_path = load_json(dp)
    lt, lv = split_train_val(
        list_path
    )

    td = DataSet(lt)
    op = (
        '/scratch/zyang195/dataset'
        '/shapenet/voxsim/'
        'lmdb/train_64_2.lmdb'
    )
    dataset2lmdb(td, op, 1000)

    vd = DataSet(lv)
    op = (
        '/scratch/zyang195/dataset'
        '/shapenet/voxsim/'
        'lmdb/val_64_2.lmdb'
    )
    dataset2lmdb(vd, op, 1000)


if __name__ == '__main__':
    main()
