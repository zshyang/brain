from parse import parse


def m2obj(m_file, obj_file):
    """Convert m file to obj file.

    Args:
        m_file:
        obj_file:
    """
    verts = []
    faces = []
    with open(m_file, "r") as file:
        for line in file:
            if line[0] == "V":
                vert = parse("Vertex {} {:f} {:f} {:f}", line)
                verts.append([vert[1], vert[2], vert[3]])
            if line[0] == "F":
                face = parse("Face {} {:d} {:d} {:d}", line)
                faces.append([face[1], face[2], face[3]])
    with open(obj_file, "w") as file:
        for vert in verts:
            file.write("v {} {} {}\n".format(vert[0], vert[1], vert[2]))
        for face in faces:
            file.write("f {} {} {}\n".format(face[0], face[1], face[2]))

