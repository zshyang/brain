"""
Author: Vinit V. Singh
Collection of utility function for mesh I/O and validity checks
TODO: Add NoneType and other validity checks and better decriptions
"""
import numpy as np
import open3d as o3d
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

def pytorch3D_mesh(f_path, device):
    """
    Read pytorch3D mesh from path

    Args:
        f_path: obj file path

    Returns:
        mesh, faces, verts, verts_normals, edges: pytorch3D mesh and other mesh information
    """
    if not f_path.endswith('.obj'):
        raise ValueError('Input files should be in obj format.')
    verts, faces, aux = load_obj(f_path, device)
    faces_idx = faces.verts_idx
    mesh = Meshes(verts=[verts], faces=[faces_idx])
    faces = mesh.faces_packed()
    verts = mesh.verts_packed()
    verts_normals = mesh.verts_normals_packed()
    edges = mesh.edges_packed()
    return mesh, faces, verts, verts_normals, edges


def open3D_mesh(faces, verts, verts_normals, edges):
    """
    Contruct Open3D mesh from faces, vertices, vertex normals, and edges

    Args:
        faces, verts, verts_normals, edges: mesh information

    Returns:
        mesh, faces, verts, verts_normals, edges: open3D mesh and other mesh information
    """
    faces = o3d.utility.Vector3iVector(np.array(faces))
    verts = o3d.utility.Vector3dVector(np.array(verts))
    verts_normals = o3d.utility.Vector3dVector(np.array(verts_normals))
    edges = o3d.utility.Vector2iVector(edges)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = verts
    mesh.triangles = faces
    mesh.vertex_normals = verts_normals
    return mesh, faces, verts, verts_normals, edges

def is_manifold(mesh):
    """
    Check if mesh is non-manifold

    Args:
        mesh: open3D mesh

    Returns:
        is_mfld: Boolean value signifying is mesh is manifold
    """
    is_mfld = True
    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    verts_manifold = mesh.is_vertex_manifold()
    if not edge_manifold or not verts_manifold or not edge_manifold_boundary:
        is_mfld = False
    return is_mfld

def visualizeMesh(mesh, verts, edges):
    """
    Utility function to visualize open3D mesh
    Args:
        mesh, verts, edges: open3D mesh and other mesh information
    """
    black = np.asarray([0, 0, 0]).astype(np.float64)
    black = np.stack([black]* len(edges), 0)
    edge_set = o3d.geometry.LineSet()
    edge_set.points = verts
    edge_set.lines = edges
    edge_set.colors = o3d.utility.Vector3dVector(black)
    o3d.visualization.draw_geometries([mesh, edge_set])

def visualizeMeshPcd(mesh, verts, edges, points):
    """
    Utility function to visualize open3D mesh along with point cloud
    Args:
        mesh, verts, edges: open3D mesh and other mesh information
        points: point cloud
    """
    black = np.asarray([0, 0, 0]).astype(np.float64)
    black = np.stack([black]* len(edges), 0)
    red = np.asarray([255, 0, 0]).astype(np.float64)
    red = np.stack([red]* len(points), 0)

    edge_set = o3d.geometry.LineSet()
    edge_set.points = verts
    edge_set.lines = edges
    edge_set.colors = o3d.utility.Vector3dVector(black)

    pcd = o3d.geometry.PointCloud()
    pcd.points = points
    pcd.colors = o3d.utility.Vector3dVector(red)
    o3d.visualization.draw_geometries([mesh, edge_set, pcd])
