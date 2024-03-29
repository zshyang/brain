''' validate the the shared data with me are correct 

author:
    Zhangsihao Yang

logs:
    2023-02-13: init
'''
import os
from glob import glob

from parse import parse


def make_file_folder(file_path):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)


def parse_vertex(line, verts):
    '''parse the vertex line
    '''
    if line[0] == 'V':
        vert = parse('Vertex {:d} {} {} {} {}', line)
        verts.append([vert[1], vert[2], vert[3]])
    return verts


def parse_face(line, faces):
    '''parse the face line
    '''
    if line[0] == 'F':
        face = parse('Face {} {:d} {:d} {:d}{}', line)
        faces.append([face[1], face[2], face[3]])
    return faces


def load_m_file(m_file_path):
    '''load the m file into verts and faces
    '''
    verts = []
    faces = []
    with open(m_file_path, 'r') as file:
        for line in file:
            verts = parse_vertex(line, verts)
            faces = parse_face(line, faces)
    return verts, faces


def save_obj(obj_file_path, verts, faces):
    ''' save the verts and faces into obj file
    '''
    make_file_folder(obj_file_path)
    with open(obj_file_path, 'w') as file:
        for vert in verts:
            file.write(f'v {vert[0]} {vert[1]} {vert[2]}\n')
        for face in faces:
            file.write(f'f {face[0]} {face[1]} {face[2]}\n')


def main():
    # verfiy that all is correct.
    all_folder_path = glob('/workspace/data/all/*/*.m')

    # save the m file into obj file.
    for m_file_path in all_folder_path:
        verts, faces = load_m_file(m_file_path)
        save_obj('validate.obj', verts, faces)
        break

    # verify the vent_out folder is correct.
    vent_out_folder_path = glob('/workspace/data/vent_out/*.m')
    # save the m file into obj file.
    for m_file_path in vent_out_folder_path:
        verts, faces = load_m_file(m_file_path)
        save_obj('validate_vent.obj', verts, faces)
        break

    return 0


if __name__ == '__main__':
    main()
