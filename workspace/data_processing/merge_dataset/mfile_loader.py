from parse import parse


class MFileLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.verts, self.faces = self.__load()

    def __load(self):
        '''load the m file into verts and faces
        '''
        verts = []
        faces = []
        with open(self.file_path, 'r') as file:
            for line in file:
                verts = self.__parse_vertex(line, verts)
                faces = self.__parse_face(line, faces)
        return verts, faces

    def __parse_vertex(self, line, verts):
        '''parse the vertex line
        '''
        if line[0] == 'V':
            vert = parse('Vertex {:d} {} {} {} {}', line)
            verts.append([vert[1], vert[2], vert[3]])
        return verts

    def __parse_face(self, line, faces):
        '''parse the face line
        '''
        if line[0] == 'F':
            face = parse('Face {} {:d} {:d} {:d}{}', line)
            faces.append([face[1], face[2], face[3]])
        return faces
