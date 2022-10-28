from parse import parse
from glob import glob
import os


def make_file_folder(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def m2obj(m_file, obj_file):
    """Convert m file to obj file.

    Args:
        m_file:
        obj_file:
    """
    # parse .m file
    verts = []
    faces = []
    with open(m_file, "r") as file:
        for line in file:
            if line[0] == "V":
                vert = parse("Vertex {} {:f} {:f} {:f} {}", line)
                verts.append([vert[1], vert[2], vert[3]])
            if line[0] == "F":
                face = parse("Face {} {:d} {:d} {:d}{}", line)
                faces.append([face[1], face[2], face[3]])

    # write the obj file
    make_file_folder(obj_file)
    with open(obj_file, "w") as file:
        for vert in verts:
            file.write("v {} {} {}\n".format(vert[0], vert[1], vert[2]))
        for face in faces:
            file.write("f {} {} {}\n".format(face[0], face[1], face[2]))


def main():
    print(glob('../MMS/AD_pos/l/*.m'))
    list_m_file_path = glob('../MMS/*/l/*.m')
    for m_file_path in list_m_file_path:
        obj_file_path = m_file_path.replace('MMS', 'obj')
        obj_file_path = obj_file_path.replace('.m', '.obj')
        m2obj(m_file_path, obj_file_path)
        break

    pass


if __name__ == '__main__':
    main()
