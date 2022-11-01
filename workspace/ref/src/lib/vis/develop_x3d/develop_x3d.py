""" 
Author: Zhangsihao.Yang
"""
import x3d
import mesh
import numpy as np
# import trimesh
# import tqdm
import json
import os

def test_draw_plane():
    split = (5, 5, 5)
    plane_ratio = 0.7
    loaded_mesh = mesh.read_obj('m07.obj')

    mesh_viewer = x3d.MeshViewer()
    mesh_color = (0, 1, 1, 1.0)
    plane_color = (0, 0, 1, 0.1)
    js_file_name = 'tmp.js'

    mesh_viewer.render_mesh_plane(
        loaded_mesh, mesh_color, split,
        plane_color, plane_ratio,
        js_file_name)


def save_single_mesh():
    """ Only save colored mesh.
    """
    loaded_mesh = mesh.read_obj('m08.obj')

    mesh_viewer = x3d.MeshViewer()
    mesh_color = (0.8, 0.8, 0.8, 1.0)

    mesh_viewer.save_colored_meshes_as_js(
        [[loaded_mesh], [mesh_color]], 'tmp.js')


def load_points(points_fn):
    points = np.loadtxt(points_fn, delimiter=' ')
    return points


def load_labels(label_fn):
    labels = np.loadtxt(label_fn, delimiter=' ')
    return labels


def assign_label_to_vertices(points, vertices, labels):
    vertex_labels_fn = 'vertex_labels.json'
    if os.path.exists(vertex_labels_fn):
        with open(vertex_labels_fn, 'r') as file:
            vertex_labels = json.load(file)
            vertex_labels = np.array(vertex_labels)
    else:
        vertex_labels = []
        for vertex in tqdm.tqdm(vertices):
            dist = mesh.distance_between_two_point_clouds(vertex, points)
            label_index = np.argmin(dist)
            label = labels[label_index]
            vertex_labels.append(int(label))

        with open(vertex_labels_fn, 'w') as file:
            json.dump(vertex_labels, file)

        vertex_labels = np.array(vertex_labels)

    return vertex_labels


def convert_label_from_vertices_to_faces(faces, vertex_labels):
    faces = np.array(faces)
    face_labels = vertex_labels[faces]
    first_ = face_labels[:, 0]
    second_ = face_labels[:, 1]
    third_ = face_labels[:, 2]
    face_labels = ((first_ == second_) * first_ + (first_ != second_) * third_)
    return second_


def convert_label_to_color(labels):
    color_base = np.array([
        [173./256, 216./256, 230./256],
        [1., 0.714, 0.757],
        [0.565, 0.933, 0.565]])
    colors = color_base[labels - 2]

    return colors


def test_draw_block():
    return


def save_segmentation():
    loaded_mesh = mesh.read_obj('m08.obj')
    points = load_points('08.pts')
    points = align_points_mesh(points, loaded_mesh)
    labels = load_labels('08.seg')
    vertices, faces = loaded_mesh


    # print(labels.shape)
    # colors = convert_label_to_color(labels.astype(int))
    # a_channel = np.ones([colors.shape[0], 1])
    # colors_rgba = np.concatenate((colors, a_channel), -1)
    # point_cloud = trimesh.points.PointCloud(points, colors=colors_rgba)
    # point_cloud.show()


    mesh_viewer = x3d.MeshViewer()

    # get_item_visualization(points, loaded_mesh)

    vertex_labels = assign_label_to_vertices(points, vertices, labels)

    face_labels = convert_label_from_vertices_to_faces(faces, vertex_labels)
    colors = convert_label_to_color(face_labels)

    # colors = convert_label_to_color(vertex_labels)

    a_channel = np.ones([colors.shape[0], 1])
    colors_rgba = np.concatenate((colors, a_channel), -1)

    mesh_viewer.save_colored_meshes_as_js(
            [[loaded_mesh], [colors_rgba]], 'tmp.js')


def get_vertices_center(vertices):
    """ Get the center of the vertices."""
    center = (vertices.min(0) + vertices.max(0)) / 2.0
    return center


def align_points_mesh(points, loaded_mesh):
    """ Align the points and mesh.
    """
    # Exchange the axis.
    points[:, [0, 2]] = points[:, [2, 0]]

    # Negative.
    points[:, 2] = -points[:, 2]
    points[:, 0] = -points[:, 0]

    # Move.
    mesh_center = get_vertices_center(loaded_mesh[0])
    point_center = get_vertices_center(points)
    points = points + point_center + mesh_center

    return points


def get_item_visualization(points, loaded_mesh):
    """ Visualize the points and the mesh.
    """
    scene = trimesh.scene.Scene()
    p_mesh = trimesh.points.PointCloud(points)
    scene.add_geometry(p_mesh)
    real_mesh = trimesh.Trimesh(
       vertices=loaded_mesh[0], faces=loaded_mesh[1])
    scene.add_geometry(real_mesh)
    scene.show()


def test_draw_block_with_attention():
    """ Draw the block with attention on different block.
    """
    are_blocks_visible = True
    key_blocks_visible = False
    important_blocks_visible = True
    list_important_blocks = [62,81,116]
    explode_scale = 1.0
    split = (5, 5, 5)

    loaded_mesh = mesh.read_obj('LHippo_60k.obj')
    vertices = loaded_mesh[0]
    vertices = vertices - np.expand_dims(
        np.mean(vertices, axis=0), 0)  # center
    dist = np.max(np.sqrt(
        np.sum(vertices ** 2, axis=1)), 0)
    vertices = vertices / dist  # scale

    loaded_mesh = (vertices, loaded_mesh[1])
        
    meshes, voxel_centers = mesh.split_mesh(loaded_mesh, split)

    list_key_blocks = [
        i for i in range(len(meshes)) if len(meshes[i][0]) > 0]
    print(list_key_blocks)

    for i in range(len(meshes)):
        if i != 116:
            meshes[i] = [[], []]


    # Make some blocks empty.
    # meshes[27] = ([], [])
    # meshes[72] = ([], [])
    # meshes[98] = ([], [])
    # meshes[53] = ([], [])
    # meshes[62] = ([], [])
    # meshes[78] = ([], [])
    # meshes[73] = ([], [])

    mesh_viewer = x3d.MeshViewer()

    voxel_centers = mesh.create_space_points(split)

    mesh_viewer.render_mesh_block(meshes, voxel_centers, split, explode_scale,
        'tmp.js', list_important_blocks, list_key_blocks, 
        are_blocks_visible, key_blocks_visible,
        important_blocks_visible)


def make_figure_2():
    """ Make the mesh for figure 2.
    """
    are_blocks_visible = True
    key_blocks_visible = False
    important_blocks_visible = True
    list_important_blocks = [85]
    explode_scale = 0.3
    split = (5, 5, 5)

    loaded_mesh = mesh.read_obj('code/develop_x3d/d01_01.obj')

    meshes, voxel_centers = mesh.split_mesh(loaded_mesh, split)

    list_key_blocks = [
        i for i in range(len(meshes)) if len(meshes[i][0]) > 0]
    print(list_key_blocks)

    # Clean list.
    for i in range(len(meshes)):
        if i != 85:
            meshes[i] = [[], []]

    mesh_viewer = x3d.MeshViewer()

    voxel_centers = mesh.create_space_points(split)

    mesh_viewer.render_mesh_block(meshes, voxel_centers, split, explode_scale,
        'code/develop_x3d/tmp.js', list_important_blocks, list_key_blocks, 
        are_blocks_visible, key_blocks_visible,
        important_blocks_visible)


def visualize_single_point_cloud():

    with open('05pc.json', 'r') as file:
        points = json.load(file)
    points = np.array(points)

    colors = np.ones_like(points)
    a_channel = np.ones([colors.shape[0], 1]) * 0.9
    colors_rgba = np.concatenate((colors, a_channel), -1)

    point_cloud = trimesh.points.PointCloud(points, colors=colors_rgba)
    point_cloud.show()


def make_figure_3_b():

    with open('code/develop_x3d/data/pc.json', 'r') as file:
        points = json.load(file)
    points = np.array(points)

    colors = np.ones_like(points)
    a_channel = np.ones([colors.shape[0], 1]) * 0.9
    colors_rgba = np.concatenate((colors, a_channel), -1)

    point_cloud = trimesh.points.PointCloud(points, colors=colors_rgba)
    point_cloud.show()


def make_figure_3_a():
    """ Make figure a in figure 3.
    """
    are_blocks_visible = True
    key_blocks_visible = False
    important_blocks_visible = True
    list_important_blocks = [118, 17]
    explode_scale = 0.3
    split = (5, 5, 5)

    loaded_mesh = mesh.read_obj('code/develop_x3d/data/air.obj')

    meshes, voxel_centers = mesh.split_mesh(loaded_mesh, split)

    list_key_blocks = [
        i for i in range(len(meshes)) if len(meshes[i][0]) > 0]
    print(list_key_blocks)

    # Clean list.
    # for i in range(len(meshes)):
    #     if i != 37:
    #         meshes[i] = [[], []]

    mesh_viewer = x3d.MeshViewer()

    voxel_centers = mesh.create_space_points(split)

    mesh_viewer.render_mesh_block(meshes, voxel_centers, split, explode_scale,
        'code/develop_x3d/tmp.js', list_important_blocks, list_key_blocks, 
        are_blocks_visible, key_blocks_visible,
        important_blocks_visible)


if __name__ == '__main__':
    # test_draw_plane()

    # save_single_mesh()

    # save_segmentation()

    # visualize_single_point_cloud()

    # test_draw_block()

    test_draw_block_with_attention()

    # make_figure_2()

    # make_figure_3_a()
    
    # make_figure_3_b()
