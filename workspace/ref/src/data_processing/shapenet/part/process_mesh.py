"""
Data is pre-processed to obtain the following infomation:
    1) Vertices and Faces of the mesh
    2) 1 Ring, 2 Ring, and 3 Ring neighborhood of the mesh faces

logs:
    20220917
        the output of running this file
  0%|                                                  | 0/5324 [00:00<?, ?it/s]skip /datasets/shapenet/part/mesh/aligned/03001627/98a1f8651c962402492d9da2668ec34c/aligned.obj
  8%|███▍                                    | 450/5324 [00:24<04:59, 16.30it/s]skip /datasets/shapenet/part/mesh/aligned/03001627/64ead031d2b04ef0504721639e19f609/aligned.obj
 10%|████                                    | 543/5324 [00:30<04:45, 16.74it/s]skip /datasets/shapenet/part/mesh/aligned/03001627/4a783b2ae8fba8b29dcf2183c858e6e5/aligned.obj
 18%|███████                                 | 942/5324 [00:54<04:22, 16.68it/s]skip /datasets/shapenet/part/mesh/aligned/03001627/b117b01ab380362db8134b0fbf68257d/aligned.obj
 22%|████████▌                              | 1169/5324 [01:08<04:05, 16.95it/s]skip /datasets/shapenet/part/mesh/aligned/03001627/f39d0db80cf41db9820bd717b66cebfc/aligned.obj
 23%|████████▊                              | 1210/5324 [01:10<04:02, 16.98it/s]skip /datasets/shapenet/part/mesh/aligned/03001627/6b32d3a9198f8b03d1dcc55e36186e4e/aligned.obj
 42%|████████████████▏                      | 2213/5324 [02:09<03:03, 16.97it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/3fd1bff496b369f71765540024eb9fef/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/03790512/6819949f5625ca12d0f568c31c1cd62a/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/03790512/6ad8d3973ccf496370a48b7db120f9fc/aligned.obj
 42%|████████████████▏                      | 2218/5324 [02:10<02:17, 22.51it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/5bd41c7d3e158ac93ff4d2f5a7608a24/aligned.obj
 42%|████████████████▎                      | 2221/5324 [02:10<02:15, 22.84it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/5bb3597d49c58017b37378f3c85478b4/aligned.obj
 42%|████████████████▎                      | 2224/5324 [02:10<02:14, 23.01it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/8ed4bdaf0c8b88ea8b31e74d456742c7/aligned.obj
 42%|████████████████▎                      | 2227/5324 [02:10<02:13, 23.27it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/47054c1839830834a88e8cb97b773125/aligned.obj
 42%|████████████████▎                      | 2230/5324 [02:10<02:12, 23.29it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/80011e85cd42668ad373c34785838ee4/aligned.obj
 42%|████████████████▍                      | 2236/5324 [02:10<02:26, 21.05it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/4548d86cf7f1c11ad373c34785838ee4/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/03790512/9f9de88a95b56660b37378f3c85478b4/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/03790512/fdb6223c286cb653cc9e7530f9d8e186/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/03790512/b649be9c09e2b332429f1c522640e6f0/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/03790512/40d84e407c46e8d8b31e74d456742c7/aligned.obj
 42%|████████████████▍                      | 2242/5324 [02:10<01:43, 29.81it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/9e9300a6e1caec217395d58407f193ba/aligned.obj
 42%|████████████████▍                      | 2246/5324 [02:11<01:55, 26.68it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/a9399a50fcb25209429f1c522640e6f0/aligned.obj
 42%|████████████████▍                      | 2249/5324 [02:11<02:03, 24.92it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/8134a965cc0b134bb37378f3c85478b4/aligned.obj
 42%|████████████████▌                      | 2255/5324 [02:11<02:19, 21.94it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/7c4fc3a05d5fc8b1d0f568c31c1cd62a/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/03790512/4a2f0b20ef680347395d58407f193ba/aligned.obj
 42%|████████████████▌                      | 2259/5324 [02:11<02:08, 23.93it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/655b9dd9425cc3a12a45a87054fa7272/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/03790512/e553d75a3635229b429f1c522640e6f0/aligned.obj
 43%|████████████████▌                      | 2263/5324 [02:11<01:59, 25.60it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/86b8e4e5ed18fe082a45a87054fa7272/aligned.obj
 43%|████████████████▌                      | 2266/5324 [02:11<02:02, 25.04it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/41cc9674e700c3fdb37378f3c85478b4/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/03790512/5037cad0901fb395b37378f3c85478b4/aligned.obj
 43%|████████████████▋                      | 2270/5324 [02:12<01:56, 26.15it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/3d37db1d974499287395d58407f193ba/aligned.obj
 43%|████████████████▋                      | 2273/5324 [02:12<02:00, 25.34it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/455485399ab75f93429f1c522640e6f0/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/03790512/365c1f92a54c8cb52a45a87054fa7272/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/03790512/49edb54e97458de8d373c34785838ee4/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/03790512/7fcee59a33976221a88e8cb97b773125/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/03790512/832c4a316c419228b37378f3c85478b4/aligned.obj
 43%|████████████████▋                      | 2279/5324 [02:12<01:32, 33.00it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/b767982d38b5171e429f1c522640e6f0/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/03790512/532e6f88a9975a27b37378f3c85478b4/aligned.obj
 43%|████████████████▋                      | 2283/5324 [02:12<01:35, 31.79it/s]skip /datasets/shapenet/part/mesh/aligned/03790512/7d75e8200565ffa7b37378f3c85478b4/aligned.obj
 47%|██████████████████▎                    | 2504/5324 [02:25<02:47, 16.79it/s]skip /datasets/shapenet/part/mesh/aligned/03467517/e66d799f51f1558a2214be36f33a634a/aligned.obj
 48%|██████████████████▋                    | 2551/5324 [02:28<02:47, 16.56it/s]skip /datasets/shapenet/part/mesh/aligned/03467517/f146c58eaa06f5e4d57700c05b1862d8/aligned.obj
 49%|██████████████████▉                    | 2590/5324 [02:30<02:43, 16.72it/s]skip /datasets/shapenet/part/mesh/aligned/03467517/5cfe03b0754039625afc17996a7c83c5/aligned.obj
 49%|██████████████████▉                    | 2593/5324 [02:30<02:30, 18.18it/s]skip /datasets/shapenet/part/mesh/aligned/03467517/482b8b9a225b6ca1d57700c05b1862d8/aligned.obj
 49%|███████████████████                    | 2604/5324 [02:31<02:35, 17.53it/s]skip /datasets/shapenet/part/mesh/aligned/03467517/8b8b084109eef6d81082f2ea630bf69e/aligned.obj
 49%|███████████████████▎                   | 2635/5324 [02:33<02:38, 17.00it/s]skip /datasets/shapenet/part/mesh/aligned/03467517/32969766a65974e9e52028751701a83/aligned.obj
 52%|████████████████████▏                  | 2752/5324 [02:40<02:32, 16.83it/s]skip /datasets/shapenet/part/mesh/aligned/02958343/6ea4111bb47039b3d1de96b5c1ba002d/aligned.obj
 52%|████████████████████▏                  | 2755/5324 [02:40<02:15, 18.91it/s]skip /datasets/shapenet/part/mesh/aligned/02958343/7434c137695be0eaf691355a196da5f/aligned.obj
 52%|████████████████████▏                  | 2763/5324 [02:40<02:18, 18.49it/s]skip /datasets/shapenet/part/mesh/aligned/02958343/b866d7e1b0336aff7c719d2d87c850d8/aligned.obj
 53%|████████████████████▌                  | 2802/5324 [02:42<02:28, 16.97it/s]skip /datasets/shapenet/part/mesh/aligned/02958343/a39ed639d1da66876d57cf36a7addb49/aligned.obj
 53%|████████████████████▋                  | 2827/5324 [02:44<02:26, 16.99it/s]skip /datasets/shapenet/part/mesh/aligned/02958343/1198255e3d20d2f323f3ca54768fe2ee/aligned.obj
 54%|█████████████████████                  | 2872/5324 [02:47<02:24, 16.95it/s]skip /datasets/shapenet/part/mesh/aligned/02958343/481c55b1fa36f6c7d834dead2eb68d68/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/02958343/f10f279643fbb3276a78cd0552215cff/aligned.obj
 54%|█████████████████████                  | 2876/5324 [02:47<01:58, 20.67it/s]skip /datasets/shapenet/part/mesh/aligned/02958343/a5dcd1196a1ffa9739f20966eb25504f/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/02958343/c2283d62367d637249b991141ee51d9a/aligned.obj
 54%|█████████████████████▏                 | 2894/5324 [02:48<02:25, 16.67it/s]skip /datasets/shapenet/part/mesh/aligned/02958343/3efdb762f2663a014c9dc258dd1682ab/aligned.obj
 54%|█████████████████████▏                 | 2897/5324 [02:48<02:09, 18.70it/s]skip /datasets/shapenet/part/mesh/aligned/02958343/5ad0fd6e9fd786937aa522cf442e862e/aligned.obj
 55%|█████████████████████▍                 | 2927/5324 [02:50<02:21, 16.99it/s]skip /datasets/shapenet/part/mesh/aligned/02958343/b87ae16029527cf3fa87597d91a3e9a2/aligned.obj
 55%|█████████████████████▌                 | 2938/5324 [02:50<02:16, 17.52it/s]skip /datasets/shapenet/part/mesh/aligned/02958343/b0a7789537663f7ba1ff2929b2f5cf19/aligned.obj
 67%|██████████████████████████             | 3559/5324 [03:27<01:43, 17.01it/s]skip /datasets/shapenet/part/mesh/aligned/04379243/8befcc7798ae971bef5d2a19d1cee3f1/aligned.obj
 71%|███████████████████████████▉           | 3806/5324 [03:41<01:31, 16.65it/s]skip /datasets/shapenet/part/mesh/aligned/04379243/15ebb1e7e6663cbfa242b893d7c243a/aligned.obj
 83%|████████████████████████████████▏      | 4401/5324 [04:17<00:56, 16.37it/s]skip /datasets/shapenet/part/mesh/aligned/04379243/14d8555f9a21f341edf3b24dfcb75e6c/aligned.obj
 87%|██████████████████████████████████     | 4650/5324 [04:31<00:39, 17.00it/s]skip /datasets/shapenet/part/mesh/aligned/03636649/aa0c8d5c38e8c87f805e3a6c310c990/aligned.obj
 88%|██████████████████████████████████▍    | 4702/5324 [04:34<00:36, 16.97it/s]skip /datasets/shapenet/part/mesh/aligned/03636649/b37e07ac31fa4f311735ea0e092a805a/aligned.obj
 89%|██████████████████████████████████▋    | 4727/5324 [04:36<00:35, 16.97it/s]skip /datasets/shapenet/part/mesh/aligned/03636649/2d638c6b6b2feb9248da169d95204ce2/aligned.obj
 89%|██████████████████████████████████▊    | 4760/5324 [04:38<00:33, 16.95it/s]skip /datasets/shapenet/part/mesh/aligned/03636649/d7465ce6bfe4b898c98f75a9ff83e3b7/aligned.obj
 91%|███████████████████████████████████▌   | 4847/5324 [04:43<00:28, 17.00it/s]skip /datasets/shapenet/part/mesh/aligned/03636649/121286f843ab37f71735ea0e092a805a/aligned.obj
 91%|███████████████████████████████████▌   | 4858/5324 [04:43<00:28, 16.53it/s]skip /datasets/shapenet/part/mesh/aligned/03636649/c54d3a5a9c8a655e46407779dbd69b2d/aligned.obj
 94%|████████████████████████████████████▍  | 4979/5324 [04:51<00:20, 16.98it/s]skip /datasets/shapenet/part/mesh/aligned/03636649/44e442591f82cd4cab0ac374f450cdc/aligned.obj
skip /datasets/shapenet/part/mesh/aligned/03636649/b15485a55d855bf980936c51aa7ffcf5/aligned.obj
 99%|██████████████████████████████████████▋| 5276/5324 [05:08<00:02, 16.96it/s]skip /datasets/shapenet/part/mesh/aligned/04099429/15474cf9caa757a528eba1f0b7744e9/aligned.obj
100%|███████████████████████████████████████| 5324/5324 [05:11<00:00, 17.10it/s]
"""
import os

import numpy as np
import torch
from pytorch3d.structures import Meshes
# from torch_geometric.data import Data
# from torch_geometric.utils import to_trimesh
from tqdm import tqdm
from trimesh.graph import face_adjacency
import trimesh as t_mesh
from utils import fpath, is_mesh_valid, normalize_mesh, pytorch3D_mesh

device = torch.device('cpu:0')
# To process the dataset enter the path where they are stored
data_root = '/datasets/shapenet/part/mesh/aligned'
max_faces = 1024
if not os.path.exists(data_root):
    raise Exception('Dataset not found at {0}'.format(data_root))

fpath_data = fpath(data_root)

for path in tqdm(fpath_data):

    if os.path.exists(path.replace('.obj', '.npz')):
        continue

    mesh, faces, verts, edges, v_normals, f_normals = pytorch3D_mesh(path, device)
    if not is_mesh_valid(mesh):
        raise ValueError('Mesh is invalid!')
    # assert faces.shape[0] == (max_faces)
    if faces.shape[0] != (max_faces):
        print(f'skip {path}')
        continue

    # Normalize Mesh
    mesh, faces, verts, edges, v_normals, f_normals = normalize_mesh(verts=verts, faces=faces)

    ########################################################################### 1st-Ring ###########################################################################
    # data = Data(pos=verts, edge_index=edges.permute(1, 0), face=faces.permute(1, 0))
    # trimesh = to_trimesh(data)
    trimesh = t_mesh.Trimesh(vertices=verts, faces=faces, process=False)

    # Neighbor faces index along edges, Edges along neighbor_faces
    faces_adjacency, edges_adjacency = face_adjacency(faces=faces.permute(1, 0),
                                                      mesh=trimesh,
                                                      return_edges=True)

    faces_neighbor_1st_ring = []
    edges_neighbor_1ring = []

    # For each face get 1-Ring neighborhood along its edges
    # For each face get edge between face and neighbor faces
    for face_idx in range(max_faces):
        face_dim_0 = np.argwhere(faces_adjacency[:, 0] == face_idx)
        face_dim_1 = np.argwhere(faces_adjacency[:, 1] == face_idx)

        face_neighbor_dim_0 = faces_adjacency[:, 0][face_dim_1]
        face_neighbor_dim_1 = faces_adjacency[:, 1][face_dim_0]

        face_neighbor_1st_ring = np.concatenate([face_neighbor_dim_0,
                                                 face_neighbor_dim_1])

        # Edge between face and neighbor faces
        face_edge = np.concatenate([face_dim_0, face_dim_1]).reshape(-1)
        edge_neighbor_1ring = edges_adjacency[face_edge]

        faces_neighbor_1st_ring.insert(face_idx, face_neighbor_1st_ring)
        edges_neighbor_1ring.insert(face_idx, edge_neighbor_1ring)

    faces_neighbor_1st_ring = np.asarray(faces_neighbor_1st_ring).squeeze(2)
    edges_neighbor_1ring = np.asarray(edges_neighbor_1ring)

    # Each face is connected to 3 other faces in the 1st Ring
    assert faces_neighbor_1st_ring.shape == (max_faces, 3)
    # Each face has 1 edge between itself and neighbor faces
    # 2 in last dim since each edge is composed of 2 vertices
    assert edges_neighbor_1ring.shape == (max_faces, 3, 2)

    ########################################################################### 2nd-Ring ###########################################################################
    faces_neighbor_0th_ring = np.arange(max_faces)
    faces_neighbor_2ring = faces_neighbor_1st_ring[faces_neighbor_1st_ring]
    faces_neighbor_0ring = np.stack([faces_neighbor_0th_ring]*3, axis=1)
    faces_neighbor_0ring = np.stack([faces_neighbor_0ring]*3, axis=2)

    dilation_mask = faces_neighbor_2ring != faces_neighbor_0ring
    faces_neighbor_2nd_ring = faces_neighbor_2ring[dilation_mask]
    faces_neighbor_2nd_ring = faces_neighbor_2nd_ring.reshape(max_faces, -1)

    # For each face there are 6 neighboring faces in its 2-Ring neighborhood
    assert faces_neighbor_2nd_ring.shape == (max_faces, 6)

    ########################################################################### 3rd-Ring ###########################################################################
    faces_neighbor_3ring = faces_neighbor_2nd_ring[faces_neighbor_1st_ring]
    faces_neighbor_3ring = faces_neighbor_3ring.reshape(max_faces, -1)

    faces_neighbor_3rd_ring = []
    for face_idx in range(max_faces):
        face_neighbor_3ring = faces_neighbor_3ring[face_idx]
        for neighbor in range(3):
            face_neighbor_1st_ring = faces_neighbor_1st_ring[face_idx, neighbor]
            dilation_mask = np.delete(
                np.arange(face_neighbor_3ring.shape[0]),
                np.where(face_neighbor_3ring == face_neighbor_1st_ring)[0][0:2])
            face_neighbor_3ring = face_neighbor_3ring[dilation_mask]
        faces_neighbor_3rd_ring.insert(face_idx, face_neighbor_3ring)
    # For each face there are 12 neighboring faces in its 3-Ring neighborhood
    faces_neighbor_3rd_ring = np.array(faces_neighbor_3rd_ring)
    assert faces_neighbor_3rd_ring.shape == (max_faces, 12)

    corners = verts[faces.long()]
    # Each face is connected to 3 other faces in the 1st Ring
    assert corners.shape == (max_faces, 3, 3)

    centers = torch.sum(corners, axis=1)/3
    assert centers.shape == (max_faces, 3)

    corners = corners.reshape(-1, 9)
    assert f_normals.shape == (max_faces, 3)

    faces_feature = np.concatenate([centers, corners, f_normals], axis=1)
    assert faces_feature.shape == (max_faces, 15)

    np.savez(path.replace('.obj', '.npz'),
             verts=verts,
             faces=faces,
             ring_1=faces_neighbor_1st_ring,
             ring_2=faces_neighbor_2nd_ring,
             ring_3=faces_neighbor_3rd_ring)

    # print(path)
    # break
