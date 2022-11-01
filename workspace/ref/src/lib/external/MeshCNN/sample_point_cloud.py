from glob import glob

import pytorch3d.io
import pytorch3d.ops
import pytorch3d.structures
from tqdm import tqdm
import os
# verts, faces, aux = pytorch3d.io.load_obj('/dataset/shrec16/shrec_16/alien/train/T9.obj')
# meshes = pytorch3d.structures.Meshes(verts=[verts], faces=[faces.verts_idx])
# sampled = pytorch3d.ops.sample_points_from_meshes(meshes)
# print(sampled)

import matplotlib.pyplot as plt
# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
from sklearn.decomposition import PCA

# a = sampled.numpy()[:, :, 0].reshape(-1)
# b = sampled.numpy()[:, :, 1].reshape(-1)
# c = sampled.numpy()[:, :, 2].reshape(-1)
# print(a.shape)
# density = a * b % 1

import numpy as np


def compute_pca(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    V = pca.components_.T
    return V

# Y = np.c_[a, b, c]
# print(Y.shape)

# V = compute_pca(Y)

# # print(V)
# print(verts.shape)
# verts = verts @ V

# print(faces)
# # pytorch3d.io.save_obj('tmp.obj', verts, faces.verts_idx)

def load_pca_save(mesh_file_name, save_path):
    verts, faces, aux = pytorch3d.io.load_obj(mesh_file_name)
    meshes = pytorch3d.structures.Meshes(verts=[verts], faces=[faces.verts_idx])
    # sample 10000 points
    sampled = pytorch3d.ops.sample_points_from_meshes(meshes)

    V = compute_pca(sampled.numpy()[0])

    verts = verts @ V

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pytorch3d.io.save_obj(save_path, verts, faces.verts_idx)


list_mesh_file_name = glob('/dataset/shrec16/shrec_16/*/*/*.obj')

print(len(list_mesh_file_name))

# save_folder = '/dataset/shrec16/pca_aligned/'

def generate_save_path(mesh_file_name):
    '''

    name convention:
        smfn = split mesh file name
    '''
    smfn = mesh_file_name.split('/')
    smfn[3] = 'pca_aligned'
    return '/'.join(smfn)


for mesh_file_name in tqdm(list_mesh_file_name):
    save_path = generate_save_path(mesh_file_name)
    load_pca_save(mesh_file_name, save_path)







# def plot_figs(fig_num, elev, azim):
#     fig = plt.figure(fig_num, figsize=(4, 3))
#     plt.clf()
#     ax = fig.add_subplot(111, projection="3d", elev=elev, azim=azim)
#     ax.set_position([0, 0, 0.95, 1])

#     ax.scatter(a[::10], b[::10], c[::10], c=density[::10], marker="+", alpha=0.4)
#     Y = np.c_[a, b, c]
#     print(Y.shape)

#     # Using SciPy's SVD, this would be:
#     # _, pca_score, Vt = scipy.linalg.svd(Y, full_matrices=False)

#     pca = PCA(n_components=3)
#     pca.fit(Y)
#     V = pca.components_.T

#     x_pca_axis, y_pca_axis, z_pca_axis = 3 * V
#     print(V)
    # x_pca_plane = np.r_[x_pca_axis[:2], -x_pca_axis[1::-1]]
    # y_pca_plane = np.r_[y_pca_axis[:2], -y_pca_axis[1::-1]]
    # z_pca_plane = np.r_[z_pca_axis[:2], -z_pca_axis[1::-1]]
    # x_pca_plane.shape = (2, 2)
    # y_pca_plane.shape = (2, 2)
    # z_pca_plane.shape = (2, 2)
    # ax.plot_surface(x_pca_plane, y_pca_plane, z_pca_plane)
    # ax.w_xaxis.set_ticklabels([])
    # ax.w_yaxis.set_ticklabels([])
    # ax.w_zaxis.set_ticklabels([])


# elev = -40
# azim = -80
# plot_figs(1, elev, azim)

# elev = 30
# azim = 20
# plot_figs(2, elev, azim)

# # plt.show()
# plt.savefig('1.png')
