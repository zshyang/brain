import trimesh
import numpy as np


def process_mesh():
    pass


def load_mesh(mesh_path):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.int32)
    return vertices, faces


def load_point(point_path):
    point = []
    with open(point_path, 'r') as point_file:
        for line in point_file.readlines():
            point.append(np.fromstring(line, sep=' '))
    point = np.stack(point)
    return point


def align_mesh_with_point(tuple_mesh, point):
    # flip tuple mesh
    vertices, faces = tuple_mesh
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    vertices = np.stack((-z, y, x), axis=1)

    # find the centers of tuple mesh and point
    center_point = (point.min(0) + point.max(0)) / 2.0
    center_vertices = (vertices.min(0) + vertices.max(0)) / 2.0

    vertices = vertices - center_vertices + center_point

    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # icp = False
    # if icp:
    #     tranformation_matrix, _ = trimesh.registration.mesh_other(mesh, point)

    #     homo_vertices = np.ones((vertices.shape[0], 4))
    #     homo_vertices[:, :3] = vertices

    #     vertices = (tranformation_matrix[:3] @ homo_vertices.T).T
    
    # samples, _ = trimesh.sample.sample_surface(mesh, point.shape[0])
    # samples = np.array(samples, dtype=np.float32)
    
    # vertices = vertices - (samples.min(0) + samples.max(0)) / 2.0 + center_point

    return vertices, faces


def draw_colored_mesh(aligned_mesh, colors, save_path):
    vertices, faces = aligned_mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rgba_colors = np.ones((colors.shape[0], 4))
    print(rgba_colors.shape)
    rgba_colors[:, :3] = colors
    mesh.visual.face_colors = colors * 255
    save_string = trimesh.exchange.obj.export_obj(mesh)
    # print(save_string)
    with open(save_path, 'w') as save_file:
        save_file.write(save_string)

    # faces = faces - faces.min() + 1
    # with open(save_path, 'w') as save_file:
    #     for vertice, color in zip(vertices, colors):
    #         save_file.write(f'v {vertice[0]} {vertice[1]} {vertice[2]} {color[0]} {color[1]} {color[2]}\n')
    #     for face in faces:
    #         save_file.write(f'f {face[0]} {face[1]} {face[2]}\n')


def project_point_to_mesh(point, aligned_mesh):
    vertices, faces = aligned_mesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    _, _, face_idx = trimesh.proximity.closest_point(mesh, point)
    print(np.unique(face_idx).shape)
    print(faces.shape)

    return np.array(face_idx, np.int32)


def load_point_label(point_label_path):
    point_label = []
    with open(point_label_path, 'r') as point_label_file:
        for line in point_label_file.readlines():
            point_label.append(np.fromstring(line, sep=' '))
    point_label = np.stack(point_label)
    return point_label


def convert_label_to_color(point_label):
    red = np.array([1, 0, 0])
    blue = np.array([0, 0, 1])

    min_index = point_label.min()
    max_index = point_label.max()

    point_label = (point_label -min_index) / (max_index - min_index)

    return point_label * red + (1 - point_label) * blue


def convert_color_from_point_to_face(colors, point_to_face_index, aligned_mesh):
    vertices, faces = aligned_mesh
    
    face_color = np.zeros((faces.shape[0], 3))
    face_color[point_to_face_index] = colors
    return face_color


def convert_color_from_face_to_vertex(face_color, aligned_mesh):
    vertices, faces = aligned_mesh

    vertex_color = np.zeros((vertices.shape[0], 3))

    # faces = np.reshape(faces, (-1))

    vertex_color[faces[:,0]] = face_color
    vertex_color[faces[:,1]] = face_color
    vertex_color[faces[:,2]] = face_color

    return vertex_color


import random
    
def point_on_triangle(pt1, pt2, pt3):
    """
    Random point on the triangle with vertices pt1, pt2 and pt3.
    """
    x, y = sorted([random.random(), random.random()])
    s, t, u = x, y - x, 1 - y
    return (s * pt1[0] + t * pt2[0] + u * pt3[0],
            s * pt1[1] + t * pt2[1] + u * pt3[1])


def sample_points_on_face(vertices, face, num_sample_points):
    # pt1, pt2, pt3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
    
    # points = [point_on_triangle(pt1, pt2, pt3) for _ in range(num_sample_points)]

    v = vertices[face]

    x = np.sort(np.random.rand(2, num_sample_points), axis=0)

    return np.column_stack([x[0], x[1]-x[0], 1.0-x[1]]) @ v

    # return np.stack(points)


def compute_point_index(sampled_points_on_face, point):
    # print(sampled_points_on_face)

    x2 = np.sum(sampled_points_on_face**2, axis=1)
    y2 = np.sum(point**2, axis=1)

    xy = np.matmul(sampled_points_on_face, point.T)

    x2 = x2.reshape(-1, 1)
    dists = x2 - 2*xy + y2

    point_index = np.argmin(dists, axis=1)

    return point_index


def major_vote(point_index):
    return np.bincount(point_index).argmax()


def compute_face_to_point_index(mesh, point):
    face_to_point_index = []
    vertices, faces = mesh

    for face in faces:
        sampled_points_on_face = sample_points_on_face(vertices, face, 25)
        point_index = compute_point_index(sampled_points_on_face, point)
        index = major_vote(point_index)
        face_to_point_index.append(index)
    return np.array(face_to_point_index, dtype=np.int)


def main():
    mesh_path = '/datasets/shapenet/part/simplified_1024/model.obj'
    point_path = '/datasets/shapenet/part/shapenetcore_partanno_segmentation_benchmark_v0/02691156/points/1a04e3eab45ca15dd86060f189eb133.pts'
    point_label_path = '/datasets/shapenet/part/shapenetcore_partanno_segmentation_benchmark_v0/02691156/points_label/1a04e3eab45ca15dd86060f189eb133.seg'

    tuple_mesh = load_mesh(mesh_path)
    point = load_point(point_path)
    point_label = load_point_label(point_label_path)
    
    colors = convert_label_to_color(point_label)

    aligned_mesh = align_mesh_with_point(tuple_mesh, point)

    face_to_point_index = compute_face_to_point_index(aligned_mesh, point)
    print(face_to_point_index.shape)

    face_color = colors[face_to_point_index]

    # point_to_face_index = project_point_to_mesh(point, aligned_mesh)
    # face_color = convert_color_from_point_to_face(colors, point_to_face_index, aligned_mesh)
    # vertex_color = convert_color_from_face_to_vertex(face_color, aligned_mesh)

    draw_colored_mesh(aligned_mesh, face_color, 'tmp.obj')


if __name__ == '__main__':
    main()
