from parse import parse


class MFileLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.verts, self.faces, self.jfeatures = self.__load()

    def __load(self):
        '''load the m file into verts and faces
        '''
        verts = []
        faces = []
        jfeatures = []
        with open(self.file_path, 'r') as file:
            try:
                for line in file:
                    verts = self.__parse_vertex(line, verts)
                    faces = self.__parse_face(line, faces)
                    jfeatures = self.__parse_jfeature(line, jfeatures)
            except UnicodeDecodeError:
                assert len(verts) == len(jfeatures) == 15000, 'at least we need verts'
        return verts, faces, jfeatures

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

    def __parse_jfeature(self, line, jfeatures):
        '''parse the jfeature line
        '''
        if line[0] == 'V':
            vert = parse('Vertex {:d} {} {} {} {}', line)
            jfeature = parse("{Jfeature=({} {} {} {} {} {} {})}\n", vert[4])
            jfeatures.append([jfeature[0], jfeature[1], jfeature[2], jfeature[3], jfeature[4], jfeature[5], jfeature[6]])
        return jfeatures

    def get_vertices(self):
        '''get the vertices
        '''
        return self.verts

    def get_faces(self):
        '''get the faces
        '''
        return self.faces

    def get_jfeatures(self):
        '''get the jfeatures
        '''
        return self.jfeatures
