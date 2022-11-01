'''
Zhangsihao Yang
04/18/2022

local test:
singularity exec \
-B /home/george/George/projects/base/src/:/workspace/ \
-B /home/george/George/dataset/:/dataset/ \
/home/george/George/singularity/occo_v2.simg \
bash
cd /workspace/lib/prep_data/mesh/agave/

agave test:
interactive
module load singularity/3.8.0
singularity exec \
-B /scratch/zyang195/projects/base/src/:/workspace/ \
-B /scratch/zyang195/dataset/:/dataset/ \
/scratch/zyang195/singularity/occo_v2.simg \
bash
python /workspace/lib/prep_data/mesh/agave/process.py --index 50000

name convention: 
cp = current path
ds = dataset
f = faces
lm = list of meshes
ln = load name
lp = lib path
m = meshes
n = name
nb = number of blocks
of = open file
pc = point cloud
sj = save json
sp = save path
tm = tuple mesh
ts = tuple to be saved
v = vertices
'''
import argparse
import json
import os
import sys

import numpy as np

cp = os.path.abspath(__file__)
lp = os.path.dirname(cp)
lp = os.path.dirname(lp)
lp = os.path.dirname(lp)
lp = os.path.dirname(lp)
lp = os.path.dirname(lp)
sys.path.append(lp)

from lib.dataset.mesh_dataset import ShapeNet


def parse_cls_idx(n):
    ns = n.split('/')
    cn = ns[6]
    idx = ns[7]
    return cn, idx


def m2lm(m):
    lm = {}
    for idx, tm in m.items():
        v, f = tm
        lm[int(idx)] = (
            v.tolist(),
            f.tolist()
        )
    return lm


def load(idx):
    if idx < 43976:
        ds = ShapeNet(
            '/dataset/shapenet/',
            'train', 64, 2
        )
    elif idx < (43976 + 7762):
        ds = ShapeNet(
            '/dataset/shapenet/',
            'val', 64, 2
        )
        idx = idx - 43976
    else:
        return None, None, None, None

    m, pc, nb, n = ds[idx]
    lm = m2lm(m)
    pc = pc.tolist()
    return lm, pc, nb, n


def group_sp(cls_idx):
    return os.path.join(
        '/dataset', 'shapenet',
        'voxsim', cls_idx[0], 
        cls_idx[1], 'model.json'
    )


def sj(ts, sp):
    os.makedirs(
        os.path.dirname(sp),
        exist_ok=True
    )
    with open(sp, 'w') as of:
        json.dump(ts, of)


def ln(idx):
    if idx < 43976:
        ds = ShapeNet(
            '/dataset/shapenet/',
            'train', -1, 999999999
        )
    elif idx < (43976 + 7762):
        ds = ShapeNet(
            '/dataset/shapenet/',
            'val', -1, 9999999999
        )
        idx = idx - 43976
    else:
        return None
    n = ds[idx]
    return n


def process():
    parser = argparse.ArgumentParser(
        description=(
            'split and simplify mesh '
            'into meshes'
        )
    )
    parser.add_argument(
        '--index', type=int, 
        help='The index'
    )
    args = parser.parse_args()

    n = ln(args.index)

    if n is None:
        return
    n = parse_cls_idx(n)
    n = group_sp(n)
    if os.path.exists(n):
        return

    ts = load(args.index)

    cls_idx = parse_cls_idx(ts[3])
    sp = group_sp(cls_idx)
    sj(ts, sp)


if __name__ == '__main__':
    process()
