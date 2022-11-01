import json
import os

import networkx as nx
import numpy as np
import six
import torch
# import trimesh
from networkx.algorithms.centrality.group import group_out_degree_centrality
from numpy.core.fromnumeric import mean, reshape

import html_generator
import x3d


def read_obj(obj_path):
    """Read vertices and faces from .obj file.
    This function will reorder the vertices and faces. The principle of the 
    ordering is based on the the order of the faces. The first vertex index
    apeared in the first face will be 0.
    This function is copyied from:
    https://github.com/deepmind/deepmind-research/blob/master/polygen/data_utils.py
    I have made two changes. And I am not sure about the robustness of this 
    function. 

    Args:
        obj_path: The path of the obj file.

    Returns:
        The vertices of the mesh.
        The faces of the mesh.
    """
    vertex_list = []  # the list of the vertices
    flat_vertices_list = []
    flat_vertices_indices = {}  # to store the name and the actual index
    flat_polygons = []

    with open(obj_path) as obj_file:  # open the file
        for line in obj_file:  # iterate the obj file
            tokens = line.split()
            if not tokens:  
                # If tokens are empty, then move to next line in the file.
                continue 
            line_type = tokens[0]
            # Skip lines not starting with v or f.
            if line_type == 'v':
                vertex_list.append([float(x) for x in tokens[1:]])
            elif line_type == 'f':
                polygon = []
                for i in range(len(tokens) - 1):
                    # get the name of the vertex
                    vertex_name = tokens[i + 1].split('/')[0]
                    # The name of the vertex has been recorded before.
                    if vertex_name in flat_vertices_indices:
                        polygon.append(flat_vertices_indices[vertex_name])
                        continue
                    # The name of the vertex has not been recorded before.
                    flat_vertex = []
                    for index in six.ensure_str(vertex_name).split('/'):
                        # If the index is empty, then move to the next index in
                        # the vertex name.
                        if not index:
                            continue
                        # obj polygon indices are 1 indexed, so subtract 1 
                        # here.
                        flat_vertex += vertex_list[int(index) - 1]
                        # If it is "//", then only the first index is 
                        # meaningful. 
                        break
                    flat_vertices_list.append(flat_vertex)
                    # This is the change I have made. Because the face is start
                    # at 0. So here is a -1.
                    flat_vertex_index = len(flat_vertices_list) - 1
                    flat_vertices_indices[vertex_name] = flat_vertex_index
                    polygon.append(flat_vertex_index)
                flat_polygons.append(polygon)

    return np.array(flat_vertices_list, dtype=np.float32), flat_polygons

def load_process_mesh(mesh_obj_path, quantization_bits=8) -> dict:
    """Load obj file and process.

    Args:
        mesh_obj_path: The path to the mesh file.
        quantization_bits: The number of bits.

    Returns:
        A dictionary that contains vertices and faces.
    """
    # Load mesh.
    vertices, faces = read_obj(mesh_obj_path)

    # Transpose so that z-axis is vertical.
    vertices = vertices[:, [2, 0, 1]]

    # Translate the vertices so that bounding box is centered at zero.
    vertices = center_vertices(vertices)

    # Scale the vertices so that the long diagonal of the bounding box is equal 
    # to one.
    vertices = normalize_vertices_scale(vertices)

    # Quantize and sort vertices, remove resulting duplicates, sort and reindex 
    # faces.
    vertices, faces, _ = quantize_process_mesh(
        vertices, faces, quantization_bits=quantization_bits)

    # Flatten faces and add 'new face' = 1 and 'stop' = 0 tokens.
    faces = flatten_faces(faces)

    # Discard degenerate meshes without faces.
    return {
        'vertices': vertices,
        'faces': faces,
    }

def flatten_faces(faces):
    """Converts from list of faces to flat face array with stopping indices.
    [-1] is the stop sign of a face.
    [-2] is the stop sign of all faces.
    And all this stop sign has been added 2 in the end.

    Args:
        faces:

    Returns:
        The flatten faces
    """
    if not faces:  # faces is empty
        return np.array([0])
    else:
        list_faces = [f + [-1] for f in faces[:-1]]
        list_faces = list_faces + [faces[-1] + [-2]]
    # pylint: disable=g-complex-comprehension
    return np.array([item for sublist in list_faces for item in sublist]) + 2  

def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero.

    Args:
        vertices:

    Returns:
        centered vertices
    """
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center

def normalize_vertices_scale(vertices):
    """Scale the vertices so that the long diagonal of the bounding box is one.

    Args:
        vertices: 

    Returns:
        scaled vertices
    """
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    extents = vert_max - vert_min
    scale = np.sqrt(np.sum(extents ** 2))
    return vertices / scale

def quantize_verts(verts, n_bits=8):
    """Convert vertices in [-0.5, 0.5] to discrete values in [0, 2**n_bits - 1].

    Args:
        verts:
        n_bits:

    Returns:
        quantized vertices
    """
    min_range = -0.5
    max_range = 0.5
    range_quantize = 2 ** n_bits - 1
    verts_quantize = (verts - min_range) * range_quantize / (
            max_range - min_range)
    if isinstance(verts_quantize, np.ndarray):
        return verts_quantize.astype('int32')
    elif isinstance(verts_quantize, torch.Tensor):
        return verts_quantize.int()
    else:
        raise NotImplementedError

def quantize_process_mesh(vertices, faces, tris=None, quantization_bits=8):
    """Quantize vertices, remove resulting duplicates and reindex faces.

    Args:
        vertices:
        faces:
        tris:
        quantization_bits:

    Returns:
        vertices:
        faces:
        tris:
    """
    vertices = quantize_verts(vertices, quantization_bits)
    vertices, inv = np.unique(vertices, axis=0, return_inverse=True)

    # Sort vertices by z then y then x.
    sort_inds = np.lexsort(vertices.T)
    vertices = vertices[sort_inds]

    # Re-index faces and tris to re-ordered vertices.
    faces = [np.argsort(sort_inds)[inv[f]] for f in faces]
    if tris is not None:
        tris = np.array([np.argsort(sort_inds)[inv[t]] for t in tris])

    # Merging duplicate vertices and re-indexing the faces causes some faces to
    # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
    # sub-faces.
    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
            # Only append faces with more than two verts.
            if c_length > 2:
                d = np.argmin(c)
                # Cyclically permute faces just that first index is the smallest.
                sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
    faces = sub_faces
    if tris is not None:
        tris = np.array([v for v in tris if len(set(v)) == len(v)])

    # Sort faces by lowest vertex indices. If two faces have the same lowest
    # index then sort by next lowest and so on.
    faces.sort(key=lambda f: tuple(sorted(f)))
    if tris is not None:
        tris = tris.tolist()
        tris.sort(key=lambda f: tuple(sorted(f)))
        tris = np.array(tris)

    # After removing degenerate faces some vertices are now unreferenced.
    # Remove these.
    num_verts = vertices.shape[0]
    vert_connected = np.equal(
        np.arange(num_verts)[:, None], np.hstack(faces)[None]).any(axis=-1)
    vertices = vertices[vert_connected]

    # Re-index faces and tris to re-ordered vertices.
    vert_indices = (
            np.arange(num_verts) - np.cumsum(1 - vert_connected.astype('int')))
    faces = [vert_indices[f].tolist() for f in faces]
    if tris is not None:
        tris = np.array([vert_indices[t].tolist() for t in tris])

    return vertices, faces, tris

def face_to_cycles(face):
    """Find cycles in face.

    Args:
        face:

    Returns:

    """
    g = nx.Graph()
    for v in range(len(face) - 1):
        g.add_edge(face[v], face[v + 1])
    g.add_edge(face[-1], face[0])
    return list(nx.cycle_basis(g))

def dequantize_verts(verts, n_bits=8, add_noise=False):
    """Convert quantized vertices to floats.

    Args:
        verts: The input vertices
        n_bits: The number of bits
        add_noise: Whether to add noise or not.

    Returns:
        dequantized vertices
    """
    min_range = -0.5
    max_range = 0.5

    range_quantize = 2 ** n_bits - 1

    if isinstance(verts, np.ndarray):  # vertices is a numpy.ndarray
        verts = verts.astype('float32')
        verts = verts * (max_range - min_range) / range_quantize + min_range
        if add_noise:
            verts += np.random.uniform(size=verts.shape) * (1 / range_quantize)
    elif isinstance(verts, torch.Tensor):  # vertices is a torch.Tensor
        verts = verts.float()
        verts = verts * (max_range - min_range) / range_quantize + min_range
        if add_noise:
            # torch.rand:
            # https://pytorch.org/docs/stable/generated/torch.rand.html
            # Add the to.(device) is to make this function to work on GPU.
            verts = verts + torch.rand(verts.shape).to(verts.device) * (
                1 / float(range_quantize))
    return verts

def unflatten_faces(flat_faces):
    """Converts from flat face sequence to a list of separate faces.

    Args:
        flat_faces: 

    Returns:
        unflatten faces.
    """

    def group(seq):
        g = []
        for el in seq:
            if el == 0 or el == -1:  # two stop sign
                yield g
                g = []
            else:
                g.append(el - 1)  # the valid face index starts at 1
        yield g

    # The last g is removed. Not sure why. It seems to be redundant with the 
    # next step.
    outputs = list(group(flat_faces - 1))[:-1]
    # Remove empty faces
    # only the face with length larger than 2 is returned
    return [o for o in outputs if len(o) > 2]  

def create_space_points(split):
    """Create a even spaced voxel point centers given the split parameters.

    Args:
        split: The tuple contains the split parameters.
    """
    # A corner case.
    if split == None:
        return None

    num_x, num_y, num_z = split

    # The 3D object is in a unit sphere. So the minimum and maximum in each axis
    # is 0.5.
    space_min = -0.5
    space_max = 0.5

    # Compute the sample point on each axis.
    x_sample = np.linspace(space_min, space_max, num=num_x+2)
    x_sample = x_sample[1:-1]
    y_sample = np.linspace(space_min, space_max, num=num_y+2)
    y_sample = y_sample[1:-1]
    z_sample = np.linspace(space_min, space_max, num=num_z+2)
    z_sample = z_sample[1:-1]

    # Add the sample points together.
    space_points = np.zeros((num_x, num_y, num_z, 3))
    space_points[..., 0] = space_points[..., 0] + x_sample.reshape(-1, 1, 1)
    space_points[..., 1] = space_points[..., 1] + y_sample.reshape(1, -1, 1)
    space_points[..., 2] = space_points[..., 2] + z_sample.reshape(1, 1, -1)
    space_points = np.reshape(space_points, (-1, 3))

    return space_points

def get_face_centers(mesh):
    """Get the center of each face of the input mesh.

    Args:
        mesh: The input mesh. It is the tuple of vertices and faces.
    
    Returns:
        face_centers: The center of the faces. It is a list.
    """
    vertices, faces = mesh

    # Expand the faces with vertex's coordinates.
    face_centers = []
    for face in faces:
        face_with_cord = vertices[face]
        face_center = face_with_cord.mean(axis=0)
        face_centers.append(face_center)

    # Compute the average.
    face_centers = np.array(face_centers)

    return face_centers

def distance_between_two_point_clouds(pc1, pc2):
    """Compute the distance between two point clouds. This function takes the 
    reference from:
    https://github.com/WangYueFt/dgcnn/blob/e96a7e26555c3212dbc5df8d8875f07228e1ccc2/tensorflow/utils/tf_util.py#L638
    Note that the distance is squared distance.

    Args:
        pc1: A point cloud with shape (n1, 3).
        pc2: A point cloud with shape (n2, 3).

    Returns:
        dist: A distance map with shape (n1, n2).
    """
    # Prepare the size.
    pc1 = np.reshape(pc1, (-1, 3))
    pc2 = np.reshape(pc2, (-1, 3))

    # Compute the components to get the distance.
    pc_inner = -2. * pc1 @ pc2.T
    pc1_square = np.square(pc1).sum(-1, keepdims=True)
    pc2_square_t = np.square(pc2).sum(-1, keepdims=True).T

    # Add up the components
    dist = pc1_square + pc_inner + pc2_square_t

    return dist

def clean_order_mesh(vertices, group_faces):
    """Reorder the vertices and faces in order to remove some useless vertices.
    Clean the mesh and also reorder the mesh.

    Becuase there is only part of the vertices is useful. Remove some unused 
    vertices.

    Args:
        vertices: The numpy array of vertices.
        group_faces: The group of faces.
    
    Returns:
        vertices: The ordered vertices. It is a numpy array.
        faces: The ordered faces. It is a list.
    """
    # After removing degenerate faces some vertices are now unreferenced.
    # Remove these.
    if len(group_faces) == 0:
        return [], []
    else:
        num_verts = vertices.shape[0]  # get the number of vertices
        # Get the vertices that are connected by the faces.
        vert_connected = np.equal(
            np.arange(num_verts)[:, None],
            np.hstack(group_faces)[None]).any(axis=-1)  # (Nv,)
        # The new vertices.
        vertices = vertices[vert_connected]
        vert_indices = (
                np.arange(num_verts) - np.cumsum(
                    1 - vert_connected.astype('int')))
        vert_indices = np.cumsum(vert_connected) - 1
        faces = [vert_indices[f].tolist() for f in group_faces]

        return vertices, faces

def split_mesh_given_group_id(mesh, group_id):
    """Split the mesh given the group index.

    Args:
        mesh: The mesh contains the vertices and faces.
            vertices: A numpy array.
            faces: A list. Becuase the number vertices of each face is 
                different.
        group_id: The numpy array contains the group id each face belongs to. It
            has the shape (Nf,).

    Returns:
        meshes: The split meshes. It is a list of mesh.
    """
    vertices, faces = mesh

    # Get the list of meshes.
    meshes = []
    num_groups = group_id.max() + 1  # the total number of groups
    for group_num in range(num_groups):
        # Store the faces as an array of list.
        faces_np = np.array(faces, dtype=object)  
        group_faces = faces_np[group_id==group_num].tolist()

        # Get the new ordered vertices and faces.
        group_vertices, group_faces = clean_order_mesh(vertices, group_faces)

        meshes.append([group_vertices, group_faces])

    return meshes

def split_mesh(mesh, split=(20,30,40)):
    """Split the mesh given the split parameter.

    Args:
        mesh: The mesh contains the vertices and faces.
            vertices: A numpy array.
            faces: A list. Becuase the number vertices of each face is 
                different.
        split: It is a tuple with three integers. There are the number of x 
            split, the number of y split, and the number of z split.
    Returns:
        meshes: The list of split meshes.
        voxel_centers: The numpy array of the centers of the voxel.
    """
    # Compute distance between voxel centers and face centers. Then compute the 
    # group index for each faces.
    voxel_centers = create_space_points(split)  # (Nv, 3)
    face_centers = get_face_centers(mesh)  # (Nf, 3)
    # The distance matirx with shape (Nf, Nv)
    dist = distance_between_two_point_clouds(face_centers, voxel_centers)
    group_id = np.argmin(dist, axis=-1)  # (Nf,)

    # Split the mesh into groups.
    meshes = split_mesh_given_group_id(mesh, group_id)

    return meshes, voxel_centers

def load_split_mesh(folder_path: str, split_range=125) -> dict:
    """Load the split mesh from the given folder path.

    Args:
        folder_path (str): The path to the folder that stores json files.
        split_range (int): 

    Returns:
        meshes (dict): The dictionary that contains the meshes.
    """
    meshes = {}

    for i in range(split_range):
        json_file_name = os.path.join(folder_path, f"{i:02d}.json")

        if os.path.exists(json_file_name):
            with open(json_file_name, 'r') as file:
                data = json.load(file)
                
            # Update the meshes dictionary.
            data['v'] = np.array(data['v'])
            meshes[i] = data

    return meshes

def visualize_split_meshes(split_meshes: list, folder: str, names: list=None):
    """ Visualize a list of split meshes.

    Args:
        split_meshes (list): The list of split meshes.
        folder (str): The folder to save the split meshes.
        names (list): The list of names for each mesh. The names are used to 
            save the files.
    """
    if names is None:
        names = [f"{i}.js" for i in range(len(split_meshes))]
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, 'split_meshes'), exist_ok=True)

    # Save split meshes into js files.
    mesh_viewer = x3d.MeshViewer()
    for split_mesh, name in zip(split_meshes, names):
        mesh_viewer.save_meshes_as_js(
            split_mesh, js_file_name=os.path.join(folder, "split_meshes", name))

    # Write a json file that records the locations of the js files.
    json_file_path = os.path.abspath(os.path.join(folder, 'mesh_names.json'))
    with open(json_file_path, 'w') as file:
        json_names = []
        for name in names:
            abs_path = os.path.abspath(
                os.path.join(folder, "split_meshes", name))
            json_names.append(abs_path)
        json.dump(json_names, file)

    # Pass the json file location to generate the HTML files. 
    html_generator.generate(num_per_page=50, output_folder=folder, 
             input_file_list=f"{json_file_path}", 
             input_title_list='mesh')

def test():
    split = (4, 5, 6)
    print(create_space_points(split))

if __name__ == '__main__':
    model = test()
    print('test')


