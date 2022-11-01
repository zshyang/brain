''' make the mesh be a manifold use manifoldplus. simplify 
the manifold 

Zhangsihao Yang
03/31/2022

'''
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


if __name__ == '__main__':
    man_sim('model.obj', 'sim.obj')
