import argparse
import json
import os
import sys

import numpy as np

# import sklearn.preprocessing
# import trimesh
# from tqdm import tqdm

root_folder = '/dataset/shapenet/mansim'
input_list_mesh_file = 'imn.json'
# output_list_mesh_file = 'output_mesh_names.json'
# script_path = '/scratch/zyang195/decimate_mesh/decimation.py'

def load_path_list():
    with open(os.path.join(root_folder, input_list_mesh_file), 'r') as file:
        input_list_mesh = json.load(file)
    # with open(os.path.join(root_folder, output_list_mesh_file), 'r') as file:
    #     output_list_mesh = json.load(file)
    return input_list_mesh


def convert(ifn):
    # [] [dataset] [shapenet] [ShapeNetCore.v2] [cls] [id] 
    # [models] [model_normalized.obj]
    sifn = ifn.split('/')
    sifn[3] = 'mansim'
    ofn = '/'.join(sifn)

    return ifn, ofn


def load_pick_files(index):
    input_list_mesh = load_path_list()
    if index < len(input_list_mesh):
        ifn = input_list_mesh[index]
        ifn, ofn = convert(ifn)
        return ifn, ofn
    else:
        return None, None

import os

import trimesh

TARGET_FACE_NUM = 1000
SHPN = 3

def man(ifn, tfn, verbose=False):
    # ManifoldPlus
    manifolderplus_cmd = 'manifold'
    os.system(f'{manifolderplus_cmd} --input {ifn} --output {tfn}')
    print('Finish manifold.')


def sim(tfn, ofn, recude_pct, shpn, verbose=False):
    # Simplify the mesh.
    simplification_cmd = 'simplify'
    os.system(f'{simplification_cmd} {tfn} {ofn} {recude_pct} {shpn}')


def get_rpct(tmp_fn, verbose=False):
    # Compute the precentage to reduce.
    loaded_mesh = trimesh.load(tmp_fn, force='mesh')
    num_faces = loaded_mesh.faces.shape[0]
    reduce_pct = float(TARGET_FACE_NUM / num_faces)
    if verbose:
        print(f'Precent is {recude_pct}')
    return reduce_pct


def make_if_nonempty(tfn):
    string_path = os.path.dirname(tfn)
    if string_path != '':
        os.makedirs(string_path, exist_ok=True)


def man_sim(ifn, ofn, verbose=False):
    tfn = ofn + '.obj'
    make_if_nonempty(tfn)

    man(ifn, tfn, verbose)

    rpct = get_rpct(tfn, verbose)

    sim(tfn, ofn, rpct, SHPN, verbose)

    os.system(f'rm {tfn}')


def call_decimatation_cmd(ifn, ofn):
    man_sim(ifn, ofn)


def process():
    parser = argparse.ArgumentParser(
        description='manifold and simplification'
    )
    parser.add_argument('--index', type=int, help='The index')
    args = parser.parse_args()

    # Load the input and output list and pick the file names given the index.
    ifn, ofn = load_pick_files(args.index)

    if ifn is None:
        return

    if os.path.exists(ofn):
        return

    # Make a call to the decimatation algorithm.
    call_decimatation_cmd(ifn, ofn)


if __name__ == '__main__':
    process()
