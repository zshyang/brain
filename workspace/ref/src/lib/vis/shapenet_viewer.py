'''
Zhangsihao Yang
04/02/2022

for local case:
    docker service: mesh
    volumes:
        - /home/george/George/projects/min_modelnet/src/:/workspace/
        - /home/george/George/dataset/:/dataset/

for agave case:
    interactive
    module load singularity/3.8.0
    singularity exec \
    -B /scratch/zyang195/:/workspace/ \
    /scratch/zyang195/singularity/large-mesh.sif bash
    cd /workspace/projects/base/src/lib/vis/
    python shapenet_viewer.py --machine agave
This will take a while because it takes time to gather all the files of ShapeNet.

mp = mesh path
out_fld = output folder
npp = number per page
dpr = dataset path in real world space
lpd = link path in docker image space
cpa = convert path on agave

'''
import argparse
import os
from glob import glob

from htmlviewer.html_generator import html_generator


def convert_path(list_mp):
    for i, mp in enumerate(list_mp):
        mps = mp.split('/')
        mps.pop(0)
        mps.pop(0)
        mps.pop(0)
        list_mp[i] = '/'.join(mps)


def local():
    list_title = ['mesh', 'copy mesh']
    
    list_mp = glob('/dataset/shapenet/mansim/*/*/*/*.obj')
    convert_path(list_mp)
    print(list_mp)
    list_input = [list_mp, list_mp]

    out_fld = '/dataset/shapenet/mansim/html/'

    npp = 8

    hg = html_generator(
        list_title, list_input, out_fld, npp
    )
    hg.generate()

    dpr = '/home/george/George/dataset/shapenet/mansim/'
    lpd = '/dataset/shapenet/mansim/html/mansim'
    if not os.path.islink(lpd):
        os.system(f'ln -s {dpr} {lpd}')


def cpa(list_mp):
    ''' convert the path relative to the html files.
    '''
    for i, mp in enumerate(list_mp):
        mps = mp.split('/')
        mps.pop(0)
        mps.pop(0)
        mps.pop(0)
        mps.pop(0)
        list_mp[i] = '/'.join(mps)


def agave():
    list_title = ['mesh']

    list_mp = glob(
        '/workspace/dataset/shapenet/mansim/*/*/*/*.obj'
    )
    cpa(list_mp)
    list_input = [list_mp]

    out_fld = '/workspace/dataset/shapenet/mansim/html/'

    npp = 16

    hg = html_generator(
        list_title, list_input, out_fld, npp
    )
    hg.generate()

    dpr = '/scratch/zyang195/dataset/shapenet/mansim/'
    lpd = '/workspace/dataset/shapenet/mansim/html/mansim'
    if not os.path.islink(lpd):
        os.system(f'ln -s {dpr} {lpd}')


def main():
    parser = argparse.ArgumentParser(
        description='ShapeNet viewer'
    )
    parser.add_argument(
        '--machine', type=str, help='running machine'
    )
    args = parser.parse_args()
    if args.machine == 'local':
        local()
    elif args.machine == 'agave':
        agave()


if __name__ == '__main__':
    main()
