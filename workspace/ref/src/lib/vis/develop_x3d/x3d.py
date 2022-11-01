import numpy as np

RATIO = 0.7


class MeshViewer:
    def __init__(self, width=650, height=650) -> None:
        """
        Args:
            width (int): The width of the js file window.
            height (int): The height of the js file window.
        """
        self.width = width
        self.height = height

    def save_obj_as_js(self, vertices, faces, js_file_name):
        """Save obj as java script file.
        This function will create a java script file.
        The function obj_to_html is called in this function.

        Args:
            vertices: The vertices array.
            faces: The faces array. It doesn't have to be a triangle mesh.
            js_file_name: The java script file name.
        """
        with open(js_file_name, "w") as file:
            file.write(f"document.write(`\n")
            file.write(
                f"\t<x3d id='someUniqueId' showStat='false' showLog='false' "
                f"x='0px' y='0px' width='{self.width}px' "
                f"height='{self.height}px'>\n"
                f"\t\t<scene>\n"
                f"\t\t\t<viewpoint id='aview' centerOfRotation='0 0 0' "
                f"position='0 0 3'></viewpoint>\n"
                f"\t\t\t<transform DEF='airway1' rotation='0 1 0 0'>\n"
                )
            self.obj_to_html(vertices, faces, file)
            file.write(f"\t\t\t</transform>\n"
                    f"\t\t</scene>\n"
                    f"\t</x3d>\n"
                    )
            file.write(f"`);")

    def insert_js_into_html(self, js_file_name, html_file):
        """Insert the location of the java script file into html file.

        Args:
            js_file_name: The location of the java script file.
            html_file: The opened text file.
        """
        base_indent_level = 2
        indent = "\t"
        html_file.write(f"{indent * (base_indent_level + 0)}<script src=\""
                        f"{js_file_name}\"></script>\n")

    def obj_to_html(self, v, f, html_file):
        """Write mesh into HTML that is in the <shape> scope.
        This is just a helper function. And it could be private.

        Args:
            v: The vertices
            f: The faces
            html_file: The open file
        """
        # Since we only use this function into generating the js file,
        # we do not need to change the indent level.
        base_indent_level = 3  # The base indent level
        indent = "\t"  # The indent symbol
        html_file.write(f"{indent * (base_indent_level + 1)}<shape>\n")
        html_file.write(f"{indent * (base_indent_level + 2)}<Appearance "
                        f"DEF='App'>"
                        f"\n"
                        f"{indent * (base_indent_level + 3)}<Material "
                        f"ambientIntensity='0.0243902' "
                        f"diffuseColor='0.9 0.1 0.1' "
                        f"shininess='0.12' specularColor='0.94 0.72 0' "
                        f"transparency='0.1' /> \n"
                        f"{indent * (base_indent_level + 2)}</Appearance>\n")
        html_file.write(f"{indent * (base_indent_level + 2)}<indexedFaceSet "
                        f"creaseAngle='1' convex='false' "
                        f"solid='false' coordIndex='\n")

        # Iterate the faces
        for face in f:
            html_file.write(f"{indent * (base_indent_level + 3)}")
            for face_index in face:
                html_file.write("{} ".format(face_index))
            html_file.write("-1 \n")
        html_file.write(f"{indent * (base_indent_level + 2)}'>\n")
        html_file.write(f"{indent * (base_indent_level + 3)}<coordinate "
                        f"point='\n")

        # Iterate the vertices
        for vertex in v:
            html_file.write(f"{indent * (base_indent_level + 4)}")
            html_file.write("{} {} {} \n".format(vertex[0], vertex[1], 
            vertex[2]))
        html_file.write(f"{indent * (base_indent_level + 3)}'></coordinate>\n")
        html_file.write(f"{indent * (base_indent_level + 2)}</indexedFaceSet>\n"
                        f"{indent * (base_indent_level + 1)}</shape>\n"
                        )

    def write_head(self, file_name):
        """Write the head into the HTML file.
        This is only useful in current situation which need to be changed in 
        future.

        Args:
            file_name: The open file
        """
        file_name.write(f"<!DOCTYPE html>\n"
                        f"<html>\n"
                        f"\t<head> \n"
                        f"\t\t<meta http-equiv='Content-Type' "
                        f"content='text/html;charset=utf-8'></meta>\n"
                        f"\t\t<link rel='stylesheet' type='text/css' "
                        f"href='http://www.x3dom.org/x3dom/release/x3dom.css'>"
                        f"</link>\n"
                        f"\t\t<script type='text/javascript' "
                        f"src='http://www.x3dom.org/x3dom/release/x3dom.js'>"
                        f"</script>\n"
                        f"\t</head> \n"
                        f"\t<body>\n"
                        f"\t\t<h1> Example1 </h1>\n"
                        f"\t\t<h1> Click and drag to rotate. Scroll to zoom. "
                        f"</h1>\n"
                        )

    def write_tail(self, file_name):
        """Write the tail into the HTML file.
        This is only useful in current situation which need to be changed in 
        future.

        Args:
            file_name: The open file
        """
        file_name.write(f"\t</body>\n"
                        f"</html>\n"
                        )

    def write_obj_html(self, v, f,
        html_file_name="test.html", js_file_name="test.js"):
        """Write the obj into a html view.
        """
        self.save_obj_as_js(v, f, js_file_name)
        with open(html_file_name, "w") as file:
            self.write_head(file)
            self.insert_js_into_html(js_file_name, file)
            self.write_tail(file)

    def write_meshes_html(self, meshes, voxel_centers=None, 
        html_file_name="test.html", 
        js_file_name="test.js"):
        """Write meshes into html.
        """
        self.save_meshes_as_js(meshes, voxel_centers, js_file_name)
        with open(html_file_name, "w") as file:
            self.write_head(file)
            self.insert_js_into_html(js_file_name, file)
            self.write_tail(file)

    def save_meshes_as_js(self, meshes, voxel_centers=None, explode_scale=None,
        js_file_name="test.js"):
        """Save meshes as java script file.
        This function will create a java script file.
        The function obj_to_html is called in this function.

        Args:
            vertices: The vertices array.
            faces: The faces array.
            js_file_name: The java script file name.
        """
        with open(js_file_name, "w") as file:
            file.write(f"document.write(`\n")
            file.write(
                f"\t<x3d id='someUniqueId' showStat='false' showLog='false' "
                f"x='0px' y='0px' width='650px' height='650px'>\n"
                f"\t\t<scene>\n"
                f"\t\t\t<viewpoint id='aview' centerOfRotation='0 0 0' "
                f"position='0 0 3'></viewpoint>\n"
                f"\t\t\t<transform DEF='airway1' rotation='0 1 0 0'>\n"
                )

            if isinstance(meshes, list):
                for i, mesh in enumerate(meshes):
                    vertices, faces = mesh
                    if vertices == []:
                        continue
                    if voxel_centers is None:
                        vertices = vertices
                    else:
                        vertices = vertices + explode_scale * voxel_centers[i]
                    self.obj_to_html(vertices, faces, file)
            elif isinstance(meshes, dict):
                for key in meshes:
                    if key != "meta":
                        vertices = np.array(meshes[key]["v"])
                        faces = meshes[key]["f"]
                        if voxel_centers is None:
                            vertices = vertices
                        else:
                            vertices = vertices + explode_scale * \
                                voxel_centers[key]
                        self.obj_to_html(vertices, faces, file)
            else:
                print(f"This type {type(meshes)} is not identified!")
                exit(1)

            file.write(f"\t\t\t</transform>\n"
                    f"\t\t</scene>\n"
                    f"\t</x3d>\n"
                    )
            file.write(f"`);")

    @staticmethod
    def add_colors_to_meshes(meshes, colors):
        """ Add colors to meshes
        Args:
            meshes
                The list of meshes.
            colors
                The list of colors.
        Returns:
            meshes
                The list of meshes.
            colors
                The list of colors
        """
        assert (len(meshes) == len(colors)), 'Not equal'

        return meshes, colors

    @staticmethod
    def get_center_radius(num_split, plane_ratio):
        """ Get the list of centers and radius.
        Args:
            num_split
                The number of split.
            plane_ratio
                The ratio to scale the plane.
        """
        space_min = -0.5
        space_max = 0.5
        centers = np.linspace(
            space_min, space_max, num=num_split+1)
        centers = centers[1:-1]
        radius = np.sqrt(0.25 - centers * centers) * plane_ratio
        return centers.tolist(), radius.tolist()

    @staticmethod
    def build_x_plane_mesh(center, radii):
        v = [[center, radii, radii],
             [center, -radii, radii],
             [center, -radii, -radii],
             [center, radii, -radii]]
        f = [[0, 1, 2, 3]]
        return np.array(v), f

    @staticmethod
    def build_y_plane_mesh(center, radii):
        v = [[radii, center, radii],
             [-radii, center, radii],
             [-radii, center, -radii],
             [radii, center,-radii]]
        f = [[0, 1, 2, 3]]
        return np.array(v), f

    @staticmethod
    def build_z_plane_mesh(center, radii):
        v = [[radii, radii, center],
             [-radii, radii, center],
             [-radii, -radii, center],
             [radii, -radii, center]]
        f = [[0, 1, 2, 3]]
        return np.array(v), f

    def build_x_planes(self, centers, radius):
        planes = []
        for center, radii in zip(centers, radius):
            plane = self.build_x_plane_mesh(center, radii)
            planes.append(plane)
        return planes

    def build_y_planes(self, centers, radius):
        planes = []
        for center, radii in zip(centers, radius):
            plane = self.build_y_plane_mesh(center, radii)
            planes.append(plane)
        return planes

    def build_z_planes(self, centers, radius):
        planes = []
        for center, radii in zip(centers, radius):
            plane = self.build_z_plane_mesh(center, radii)
            planes.append(plane)
        return planes

    def construct_planes(self, centers, radius, i):
        if i == 0:
            planes = self.build_x_planes(centers, radius)
        elif i == 1:
            planes = self.build_y_planes(centers, radius)
        elif i == 2:
            planes = self.build_z_planes(centers, radius)
        else:
            raise NotImplementedError
        return planes

    def create_planes(self, split, plane_ratio):
        planes = []
        for i, num_split in enumerate(split):
            centers, radius = self.get_center_radius(
                num_split, plane_ratio)
            planes.extend(
                self.construct_planes(
                    centers, radius, i))
        return planes

    def add_planes(self, colored_meshes, plane_color, split,
        plane_ratio=0.7):
        planes = self.create_planes(split, plane_ratio)
        colors = [
            plane_color for _ in range(len(planes))]
        _, colors = self.add_colors_to_meshes(planes, colors)
        colored_planes = [planes, colors]
        return self.merge_planes_to_meshes(
            colored_meshes, colored_planes)

    @staticmethod
    def merge_planes_to_meshes(colored_meshes, colored_planes):
        """ Add colored planes to colored meshes.
        Args:
            colored_meshes
                A list of list of meshes and list of colors.
            colored_planes
                A list of list of meshes and list of colors.
        """
        meshes = []
        colors = []

        meshes.extend(colored_meshes[0])
        colors.extend(colored_meshes[1])

        meshes.extend(colored_planes[0])
        colors.extend(colored_planes[1])

        return meshes, colors

    @staticmethod
    def write_js_head(file):
        file.write(f"document.write(`\n")
        file.write(
            f"\t<x3d id='someUniqueId' showStat='false' showLog='false' "
            f"x='0px' y='0px' width='650px' height='650px'>\n"
            f"\t\t<scene>\n"
            f"\t\t\t<viewpoint id='aview' centerOfRotation='0 0 0' "
            f"position='0 0 3'></viewpoint>\n"
            f"\t\t\t<transform DEF='airway1' rotation='0 1 0 0'>\n"
        )

    def color_obj_to_html(self, vertices, faces, color, file):
        """ Write colored mesh into HTML that is in the <shape> scope.
        This is just a helper function. And it could be private.
        Args:
            vertices
                The vertices
            faces
                The faces
            color
                The color matrix for each face
            file: The open file
        """
        # Since we only use this function into generating the js file,
        # we do not need to change the indent level.
        base_indent_level = 3  # The base indent level
        indent = "\t"  # The indent symbol
        file.write(f"{indent * (base_indent_level + 1)}<shape>\n")

        file.write(f"{indent * (base_indent_level + 2)}<Appearance "
                   f"DEF='App'>"
                   f"\n")

        file.write(f"{indent * (base_indent_level + 3)}<Material "
                   f"ambientIntensity='1.00' ")

        if isinstance(color, tuple):
            file.write(f'diffuseColor="{color[0]} {color[1]} {color[2]}"')
        else:
            file.write(f'diffuseColor="0 0 1"')

        if isinstance(color, tuple):
            file.write(f"transparency='{1-color[3]}' /> \n")
        else:
            file.write(f"transparency='0.0' /> \n")

        file.write(f"{indent * (base_indent_level + 3)}</Material> \n")

        # file.write(f'<material></material>\n')

        file.write(f"{indent * (base_indent_level + 2)}</Appearance>\n")

        # Iterate the faces
        file.write(f"{indent * (base_indent_level + 2)}<indexedFaceSet "
                   f"colorPerVertex='false' creaseAngle='1' convex='false' "
                   f"coordIndex='\n")
        for face in faces:
            file.write(f"{indent * (base_indent_level + 3)}")
            for face_index in face:
                file.write("{} ".format(face_index))
            file.write("-1 \n")
        file.write(f"{indent * (base_indent_level + 2)}'solid='false' >\n")

        # Iterate the color
        if isinstance(color, np.ndarray):
            file.write(f'<colorrgba color="\n')
            for color_i in color:
                file.write(f"{color_i[0]} {color_i[1]} {color_i[2]} {color_i[3]}\n")
            file.write(f'"></colorrgba>\n')

        # Iterate the vertices
        file.write(f"{indent * (base_indent_level + 3)}<coordinate "
                   f"point='\n")
        for vertex in vertices:
            file.write(f"{indent * (base_indent_level + 4)}")
            file.write("{} {} {} \n".format(vertex[0], vertex[1], vertex[2]))
        file.write(f"{indent * (base_indent_level + 3)}'></coordinate>\n")

        # The end.
        file.write(f"{indent * (base_indent_level + 2)}</indexedFaceSet>\n"
                   f"{indent * (base_indent_level + 1)}</shape>\n")

    def write_js_list_mesh_color(self, meshes, colors, file):
        for mesh_i, color_i in zip(meshes, colors):
            vertices, faces = mesh_i
            self.color_obj_to_html(vertices, faces, color_i, file)

    @staticmethod
    def write_js_tail(file):
        file.write(f"\t\t\t</transform>\n"
                   f"\t\t</scene>\n"
                   f"\t</x3d>\n"
                   )
        file.write(f"`);")

    def save_colored_meshes_as_js(self, colored_meshes, js_file_name):
        """ Save meshes given the colors.
        Args:
            colored_meshes
                contains the following elements:
                meshes: The list of meshes.
                colors: The list of color.
            js_file_name
                The file to save the js file.
        """
        meshes, colors = colored_meshes
        with open(js_file_name, "w") as file:
            self.write_js_head(file)
            if isinstance(meshes, list):
                self.write_js_list_mesh_color(meshes, colors, file)
            self.write_js_tail(file)

    def render_mesh_plane(self, loaded_mesh, mesh_color, split,
        plane_color, plane_ratio, js_file_name):
        """ Render mesh with plane
        Args:
            loaded_mesh:
            mesh_color:
            split:
            plane_color:
            plane_ratio:
            js_file_name
                The name.
        """
        colored_meshes = self.add_colors_to_meshes(
            [loaded_mesh], [mesh_color])

        colored_meshes = self.add_planes(
            colored_meshes, plane_color, split,
            plane_ratio)

        self.save_colored_meshes_as_js(
            colored_meshes, js_file_name)
    
    def create_meta_block(self, block_length):
        half_length = block_length / 2.0
        v = [[-half_length, -half_length, half_length],
             [-half_length, half_length, half_length],
             [-half_length, -half_length, -half_length],
             [-half_length, half_length, -half_length],
             [half_length, -half_length, half_length],
             [half_length, half_length, half_length],
             [half_length, -half_length, -half_length],
             [half_length, half_length, -half_length]]

        f = [[0, 1, 3, 2],
             [2, 3, 7, 6],
             [6, 7, 5, 4],
             [4, 5, 1, 0],
             [2, 6, 4, 0],
             [7, 3, 1, 5]]

        return [np.array(v), f]

    def create_block(self, voxel_center, explode_scale, block_length):
        meta_block = self.create_meta_block(block_length)

        # Explode
        meta_block[0] = meta_block[0] + voxel_center * (1. + explode_scale)

        return meta_block

    def create_blocks(self, voxel_centers, split, explode_scale, 
        list_important_blocks, list_key_blocks, are_blocks_visible, 
        key_blocks_visible, important_blocks_visible):
        """ Create a list of blocks.
        """
        block_length = 1 / split[0]
        blocks = []
        colors = []

        if are_blocks_visible:
            alpha = 0.2
        else:
            alpha = 0.0

        for i, voxel_center in enumerate(voxel_centers):
            block = self.create_block(
                voxel_center, explode_scale, block_length)
            blocks.append(block)

            color = (0., 0., 0., 0.)

            if key_blocks_visible:
                if i in list_key_blocks:
                    color = (0.678, 0.847, 0.902, alpha)

            if important_blocks_visible:
                if i in list_important_blocks:
                    color = (1.0, 0.5, 0.5, 0.5)
            
            if (not key_blocks_visible) and (not important_blocks_visible):
                color = (0.678, 0.847, 0.902, alpha)

            colors.append(color)


        return blocks, colors

    def explode_meshes(self, meshes, explode_scale, voxel_centers):
        """ Explode meshes.
        """
        return_meshes = []
        for i, mesh in enumerate(meshes):
            voxel_center = voxel_centers[i]
            if len(mesh[0]) == 0:
                return_meshes.append(([], []))
            else:
                vertices, faces = mesh[0], mesh[1]
                vertices = vertices + explode_scale * voxel_center
                return_meshes.append((vertices, faces))
        return return_meshes

    def render_mesh_block(self, meshes, voxel_centers, split,  explode_scale,
        js_file_name, list_important_blocks, list_key_blocks,
        are_blocks_visible, key_blocks_visible,
        important_blocks_visible):
        """ Render mesh with block.
        """
        meshes = self.explode_meshes(meshes, explode_scale, voxel_centers)

        mesh_color = (0.9, 0.9, 0.9, 1.0)
        mesh_colors = [mesh_color for _ in range(len(meshes))]


        blocks, colors = self.create_blocks(
            voxel_centers, split, explode_scale, list_important_blocks, 
            list_key_blocks, are_blocks_visible, key_blocks_visible,
            important_blocks_visible)

        meshes.extend(blocks)
        mesh_colors.extend(colors)

        self.save_colored_meshes_as_js(
            [meshes, mesh_colors], js_file_name)
