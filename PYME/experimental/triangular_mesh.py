import numpy as np

class TriangularMesh(object):
    def __init__(self, vertices, faces):
        self._vertices = vertices
        self.faces = faces
        self._face_normals = None
        self._vertex_normals = None
        self._neighbors = None
        self._face_indices = None
        self._face_areas = None

        self.vertex_properties = ['x', 'y', 'z']

    def keys(self):
        return list(self.vertex_properties)

    def __getitem__(self, k):
        # this defers evaluation of the properties until we actually access them, as opposed to the mappings which
        # stored the values on class creation.
        try:
            res = getattr(self, k)
        except AttributeError:
            raise KeyError('Key %s not defined' % k)
        
        return res

    @property
    def x(self):
        return self.vertices[:, 0]

    @property
    def y(self):
        return self.vertices[:, 1]

    @property
    def z(self):
        return self.vertices[:, 2]

    @classmethod
    def from_stl(cls, filename):
        """
        Read from an STL file.
        """
        from PYME.IO.FileUtils import stl

        # Load an STL from file
        triangles_stl = stl.load_stl_binary(filename)

        # Call from_np_stl on the file stream
        return cls.from_np_stl(triangles_stl)
    
    def __copy__(self):
        return type(self)(np.copy(self.vertices), np.copy(self.faces))

    @classmethod
    def from_np_stl(cls, triangles_stl):
        """
        Read from an already-loaded STL stream.
        """
        vertices_raw = np.vstack((triangles_stl['vertex0'], 
                                  triangles_stl['vertex1'], 
                                  triangles_stl['vertex2']))
        vertices, faces_raw = np.unique(vertices_raw, 
                                        return_inverse=True, 
                                        axis=0)
        faces = faces_raw.reshape(faces_raw.shape[0] // 3, 3, order='F')

        return cls(vertices, faces)
        
    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, vertices):
        self._vertices = vertices

        #invalidate cached normals etc ...
        self._face_normals = None
        self._vertex_normals = None
        self._neighbors = None
        self._face_indices = None
        self._face_areas = None

    @property
    def face_normals(self):
        """
        Compute the normals at each face using the standard cross
        product approach.
        """
        if self._face_normals is None:
            triangles = self.vertices[self.faces]
            u = triangles[:, 2] - triangles[:, 1]
            v = triangles[:, 0] - triangles[:, 1]
            n = np.cross(u, v, axis=1)
            nn = np.linalg.norm(n, axis=1)
            self._face_areas = 0.5*nn
            self._face_normals = n/nn[:,None]
        return self._face_normals

    @property
    def face_areas(self):
        if self._face_areas is None:
            self.face_normals
        return self._face_areas

    @property
    def vertex_normals(self):
        """
        Compute the normals at each vertex by averaging face
        normal values from the surrounding vertices.
        """
        if self._vertex_normals is None:
            self._vertex_normals = np.zeros_like(self.vertices)
            # Compute vertex normals as (weighted) average of
            # face normals
            for iv in range(self.vertices.shape[0]):
                # Grab the neighboring face normals
                nb = self.face_indices[iv]
                n = self.face_normals[nb]

                # Get the neighbors for 1/r weighting
                fa = self.face_areas[nb]
                fa_sum = fa.sum()

                # Compute the mean of the face normals at the vertex
                nm = np.sum(n*fa[:,None], axis=0)/fa_sum
                nn = np.linalg.norm(nm)
                nm = nm/nn

                self._vertex_normals[iv] = nm
        return self._vertex_normals

    @property
    def neighbors(self):
        if self._neighbors is None:
            self.get_face_indices_neighbors()
        return self._neighbors

    @property
    def face_indices(self):
        if self._face_indices is None:
            self.get_face_indices_neighbors()
        return self._face_indices

    def get_face_indices_neighbors(self):
        """
        Compute mappings of vertices to neighbors and vertices
        to faces.
        """
        self._face_indices = [set()] * self.vertices.shape[0]
        self._neighbors = [set()] * self.vertices.shape[0]
        for i, f in enumerate(self.faces):
            for v in f:
                self._face_indices[v] = self._face_indices[v].union(set([i]))
                self._neighbors[v] = self._neighbors[v].union(
                                     set(f) - set([v]))
        self._face_indices = [list(f) for f in self._face_indices]
        self._neighbors = [list(n) for n in self._neighbors]

    def add_vertex(self, face_indices):
        """
        Add a vertex(es) at the centroid of self.faces[face_index].

        Parameters
        ----------
            face_indices : int or array
                Index(es) of mesh face where we wish to add a point.

        Returns
        -------
            None
        """
    
        # If we have an integer, wrap it in a list so we can loop
        if type(face_indices) == int:
            face_indices = [face_indices]
    
        # If we have an empty list, alert and exit
        if len(face_indices) == 0:
            raise ValueError('No faces to subdivide.')
    
        new_vertices = np.copy(self.vertices)
        new_faces = np.array([0, 0, 0])
    
        # Loop over all faces we're passed
        for face_index in face_indices:
            # Current faces/vertices
            f = self.faces[face_index]
            v = self.vertices[f]
        
            # New faces/vertices
            nv = np.mean(v, axis=0)
            niv = new_vertices.shape[0]
            nf = np.array([[f[0], niv, f[2]],
                           [f[0], f[1], niv],
                           [f[1], f[2], niv]])
        
            # Keep a running list of the vertices we've added
            new_vertices = np.vstack([new_vertices, nv])
            new_faces = np.vstack([new_faces, nf])
    
        # Update
        self.vertices = new_vertices
        new_face_indices = set(range(self.faces.shape[0])) - set(face_indices)
        self.faces = np.vstack([self.faces[list(new_face_indices), :], new_faces[1:, ]])

    def to_stl(self, filename):
        """
        Save list of triangles to binary STL file.

        Parameters
        ----------
            filename : string
                File name to write

        Returns
        -------
            None
        """
        from PYME.IO.FileUtils import stl

        dt = np.dtype([('normal', '3f4'), ('vertex0', '3f4'), 
                       ('vertex1', '3f4'), ('vertex2', '3f4')])

        triangles_stl = np.zeros(self.face_normals.shape[0], dtype=dt)
        triangles_stl['vertex0'] = self.vertices[self.faces[:, 0]]
        triangles_stl['vertex1'] = self.vertices[self.faces[:, 1]]
        triangles_stl['vertex2'] = self.vertices[self.faces[:, 2]]
        triangles_stl['normal'] = self.face_normals

        stl.save_stl_binary(filename, triangles_stl)

    def to_np_stl(self):
        """
        Convert mesh to numpy-stl format for viewing in a
        Jupyter notebook. This is mostly a debugging function.
        """
        from stl import mesh

        obj = mesh.Mesh(np.zeros(self.faces.shape[0], 
                        dtype=mesh.Mesh.dtype))
        for i, f in enumerate(self.faces):
            for j in range(3):
                obj.vectors[i][j] = self.vertices[f[j],:]

        return obj