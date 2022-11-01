'''
author:
    zhangsihao yang

logs:
    20220918
        file created
'''
import numpy as np
import trimesh


def save_mesh(aligned_mesh_path, aligned_mesh):
    vertices, faces = aligned_mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    save_string = trimesh.exchange.obj.export_obj(mesh)

    with open(aligned_mesh_path, 'w') as save_file:
        save_file.write(save_string)


def main():
    npz_path = '/datasets/FAUST/synthetic/shapes/tr_reg_000.npz'
    npz_file = np.load(npz_path)
    vertices, faces = npz_file['verts'], npz_file['faces']
    save_mesh('tmp.obj', (vertices, faces))


if __name__ == '__main__':
    main()
