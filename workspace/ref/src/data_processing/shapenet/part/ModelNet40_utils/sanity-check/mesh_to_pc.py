def sample_faces(vertices, faces, n_samples=10**5):
  """
  Samples point cloud on the surface of the model defined as vectices and
  faces. This function uses vectorized operations so fast at the cost of some
  memory.

  Parameters:
    vertices  - n x 3 matrix
    faces     - n x 3 matrix
    n_samples - positive integer

  Return:
    vertices - point cloud

  Reference :
    [1] Barycentric coordinate system

    \begin{align}
      P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
    \end{align}
  """
  vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                       vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
  face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))
  face_areas = face_areas / np.sum(face_areas)

  # Sample exactly n_samples. First, oversample points and remove redundant
  # Error fix by Yangyan (yangyan.lee@gmail.com) 2017-Aug-7
  n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
  floor_num = np.sum(n_samples_per_face) - n_samples
  if floor_num > 0:
    indices = np.where(n_samples_per_face > 0)[0]
    floor_indices = np.random.choice(indices, floor_num, replace=True)
    n_samples_per_face[floor_indices] -= 1

  n_samples = np.sum(n_samples_per_face)

  # Create a vector that contains the face indices
  sample_face_idx = np.zeros((n_samples, ), dtype=int)
  acc = 0
  for face_idx, _n_sample in enumerate(n_samples_per_face):
    sample_face_idx[acc: acc + _n_sample] = face_idx
    acc += _n_sample

  r = np.random.rand(n_samples, 2);
  A = vertices[faces[sample_face_idx, 0], :]
  B = vertices[faces[sample_face_idx, 1], :]
  C = vertices[faces[sample_face_idx, 2], :]
  P = (1 - np.sqrt(r[:,0:1])) * A + np.sqrt(r[:,0:1]) * (1 - r[:,1:]) * B + \
      np.sqrt(r[:,0:1]) * r[:,1:] * C
  return P

# import pymesh
# import numpy as np
# path = '../curtain_0066m.obj'
# mesh = pymesh.load_mesh(path)
# points = sample_faces(mesh.vertices, mesh.faces)
# np.savez(path.replace('.obj', '.npz'), points=points)

import numpy as np
import open3d as o3d
points = np.load('../curtain_0066m.npz')['points']
print(points.shape)
red = np.asarray([255, 0, 0]).astype(np.float64)
red = np.stack([red]* len(points), 0)

points = o3d.utility.Vector3dVector(np.array(points))
pcd = o3d.geometry.PointCloud()
pcd.points = points

o3d.io.write_point_cloud("../curtain_0066m.ply", pcd)
