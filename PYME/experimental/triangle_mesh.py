import numpy as np

def pack_edges(arr, axis=1):
    """
    Stores edges as a single unique integer. 
    
    NOTE: that this implictly caps the mesh at 2**16 
    vertices. If we need more, adjust np.uint32 to
    np.uint64 and << 16 to << 32.
    """
    arr = np.sort(arr, axis=1)
    res = ((arr[:,0].astype(np.uint32)) << 16)
    res += arr[:,1].astype(np.uint32)
    
    return res

def fast_3x3_cross(a,b):
    # Index only once
    a0 = a[0]
    a1 = a[1]
    a2 = a[2]
    b0 = b[0]
    b1 = b[1]
    b2 = b[2]

    x = a1*b2 - a2*b1
    y = a2*b0 - a0*b2
    z = a0*b1 - a1*b0

    vec = np.array([x,y,z])
    return vec

class TriangleMesh(object):
    def __init__(self, vertices, faces, max_valence=6):
        """
        Base class for triangle meshes stored using halfedge data structure. 
        Expects STL-like input.

        Parameters
        ----------
            vertices : np.array
                N x 3 array of Euclidean points.
            faces : np .array
                M x 3 array of vertex indices indicating triangle connectivity. 
                Expects STL redundancy.
        """
        self._vertices = vertices
        self._vertex_halfedges = None  # Contains a pointer to one halfedge emanating from each vertex
        self._faces = None  # Contains a pointer to one halfedge associated with each face
        self._faces_by_vertex = None  # Representation of faces by triplets of vertices

        # Halfedges
        self._h_vertex = None
        self._h_face = None
        self._h_twin = None
        self._h_next = None
        self._h_prev = None
        self._h_length = None
        self._initialize_halfedges(vertices, faces)

        # Neighbors
        self.max_valence = max_valence
        self._vertex_neighbors = None
        self._valences = None

        # Normals
        self._face_normals = None
        self._face_areas = None
        self._vertex_normals = None

        # Populate the normals
        self.face_normals
        self.vertex_normals

        # Properties we can visualize
        self.vertex_properties = ['x', 'y', 'z']

    def __getitem__(self, k):
        # this defers evaluation of the properties until we actually access them, as opposed to the mappings which
        # stored the values on class creation.
        try:
            res = getattr(self, k)
        except AttributeError:
            raise KeyError('Key %s not defined' % k)
        
        return res

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

        # triangles_returned = vertices[faces]
        # normals = np.cross((triangles_returned[:, 2] - triangles_returned[:, 1]),
        #                    (triangles_returned[:, 0] - triangles_returned[:, 1]))
        # print(normals)

        return cls(vertices, faces)

    @property
    def x(self):
        return self.vertices[:,0]
        
    @property
    def y(self):
        return self.vertices[:,1]
        
    @property
    def z(self):
        return self.vertices[:,2]

    @property
    def vertices(self):
        return self._vertices

    @property
    def faces(self):
        """
        Returns faces by vertex for rendering and STL applications.
        """
        if self._faces_by_vertex is None:
            faces = self._faces[self._faces != -1]
            v0 = self._h_vertex[self._h_prev[faces]]
            v1 = self._h_vertex[faces]
            v2 = self._h_vertex[self._h_next[faces]]
            self._faces_by_vertex = np.vstack([v0, v1, v2]).T
        return self._faces_by_vertex

    @property
    def face_normals(self):
        """
        Calculate and return the normal of each triangular face.
        """
        if self._face_normals is None:
            v2 = self.vertices[self._h_vertex[self._h_prev[self._faces]]]
            v1 = self.vertices[self._h_vertex[self._faces]]
            v0 = self.vertices[self._h_vertex[self._h_next[self._faces]]]
            u = v2 - v1
            v = v0 - v1
            n = np.cross(u, v, axis=1)
            nn = np.linalg.norm(n, axis=1)
            self._face_areas = 0.5*nn
            self._face_normals = n/nn[:, None]
            self._face_normals[np.isnan(self.face_normals)] = 0
            self._face_normals[self._faces == -1] = 0
        return self._face_normals

    @property
    def face_areas(self):
        """
        Calculate and return the area of each triangular face.
        """
        if self._face_areas is None:
            self._face_normals = None
            self.face_normals
        return self._face_areas
    
    @property
    def vertex_neighbors(self):
        """
        Return the up to self.max_valence neighbors of each vertex.
        """
        if self._vertex_neighbors is None:
            self._vertex_neighbors = -1*np.ones((len(self.vertices), self.max_valence), dtype=np.int)
            self._vertex_normals = np.zeros((len(self.vertices), 3))
            self._valences = np.zeros(len(self.vertices), dtype=np.int)
            for v_idx in np.arange(len(self.vertices)):
                _orig = self._vertex_halfedges[v_idx]
                _curr = _orig
                _twin = self._h_twin[_curr]
                i = 0
                area = 0
                self._valences[v_idx] = 0
                while True:
                    if (_curr == -1) or (_twin == -1):
                        break
                    _vertex = self._h_vertex[_curr]
                    _face = self._h_face[_curr]
                    if (i < self.max_valence):
                        self._vertex_neighbors[v_idx, i] = _curr
                        n = self.face_normals[_face]
                        a = self.face_areas[_face]
                        self._vertex_normals[v_idx] += n*a
                        area += a
                    l = self._vertices[_vertex] - self._vertices[self._h_vertex[self._h_prev[_curr]]]
                    self._h_length[_curr] = np.sqrt((l*l).sum())
                    _curr = self._h_next[_twin]
                    _twin = self._h_twin[_curr]
                    i += 1
                    self._valences[v_idx] += 1
                    if (_curr == _orig):
                        break
                self._vertex_normals[v_idx] /= area
            nn = np.linalg.norm(self._vertex_normals, axis=1)
            self._vertex_normals /= nn[:, None]
            self._vertex_normals[np.isnan(self._vertex_normals)] = 0

        return self._vertex_neighbors

    @property
    def vertex_normals(self):
        """
        Calculate and return the normal of each vertex.
        """
        if self._vertex_normals is None:
            self._vertex_neighbors = None
            self.vertex_neighbors
        return self._vertex_normals

    @property
    def valences(self):
        """
        Calculate and return the valence of each vertex.
        """
        if self._valences is None:
            self._vertex_neighbors = None
            self.vertex_neighbors
        return self._valences

    def keys(self):
        return list(self.vertex_properties)

    def _initialize_halfedges(self, vertices, faces):
        """
        Accepts unordered vertices, indices parameterization of a triangular mesh
        from an STL file and converts to topologically-connected halfedge representation.

        Parameters
        ----------
            vertices : np.array
                N x 3 array of Euclidean points.
            faces : np .array
                M x 3 array of vertex indices indicating triangle connectivity. 
                Expects STL redundancy.
        """
        if vertices is not None and faces is not None:
            # Unordered halfedges
            edges = np.vstack([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]])

            # Now order them...
            n_faces = len(faces)
            n_vertices = len(vertices)
            n_edges = len(edges)
            j = np.arange(n_faces)

            self._h_vertex = edges[:, 1]
            self._vertex_halfedges = np.zeros(n_vertices, dtype=np.int)
            self._faces = np.zeros(n_vertices, dtype=np.int)
            #self._vertices_halfedge[np.unique(edges[:, 0])] = np.arange(n_vertices)

            self._h_twin = -1*np.ones(n_edges, dtype=np.int)
            
            # for i in range(n_edges):
            #     try:
            #         self._h_twin[i] = np.argwhere(np.all(edges[i,:] == edges[:,::-1], axis=1))[0]
                    
            #         # FIXME: This sets the same entry multiple times in self._vertices_halfedge.
            #         self._vertex_halfedges[self._h_vertex[i]] = self._h_twin[i]
            #     except(IndexError):
            #         self._h_twin[i] = -1  # No twin edge exists

            # Create a unique handle for each edge
            edges_packed = pack_edges(edges)
            
            # Use a dictionary to keep track of which edges are already assigned twins
            d = {}
            for i, e in enumerate(edges_packed):
                if e in list(d.keys()):
                    idx = d.pop(e)
                    self._h_twin[idx] = i
                    self._h_twin[i] = idx
                else:
                    d[e] = i

            # Set remaining unassigned edges to -1
            for e in list(d.keys()):
                idx = d.pop(e)
                self._h_twin[idx] = -1

            self._h_face = np.hstack([j, j, j])
            self._h_next = np.hstack([j+n_faces, j+2*n_faces, j])
            self._h_prev = np.hstack([j+2*n_faces, j, j+n_faces])
            self._h_length = np.linalg.norm(self._vertices[self._h_vertex] - self._vertices[self._h_vertex[self._h_prev]], axis=1)
            
            # Set the vertex halfedges (multiassignment, but should be fine)
            self._vertex_halfedges[self._h_vertex] = self._h_next

            self._faces = j  # Faces are defined by associated halfedges
        else:
            raise ValueError('Mesh does not contain vertices and faces.')

    def _update_face_normals(self, f_idxs):
        """
        Recompute len(f_idxs) face normals.

        Parameters
        ----------
        f_idxs : list or np.array
            List of face indices to recompute.
        """

        for f_idx in f_idxs:
            v2 = self.vertices[self._h_vertex[self._h_prev[self._faces[f_idx]]]]
            v1 = self.vertices[self._h_vertex[self._faces[f_idx]]]
            v0 = self.vertices[self._h_vertex[self._h_next[self._faces[f_idx]]]]
            u = v2 - v1
            v = v0 - v1
            n = fast_3x3_cross(u.squeeze(), v.squeeze())
            # nn = np.linalg.norm(n)
            nn = np.sqrt((n*n).sum())
            self._face_areas[f_idx] = 0.5*nn
            if nn > 0:
                self._face_normals[f_idx] = n/nn
            else:
                self._face_normals[f_idx] = 0

    def _update_vertex_neighbors(self, v_idxs):
        """
        Recalculate len(v_idxs) vertex neighbors/normals.

        Parameters
        ----------
            v_idxs : list or np.array
                List of vertex indicies indicating which vertices to update.
        """
        for v_idx in v_idxs:
            _orig = self._vertex_halfedges[v_idx]
            _curr = _orig
            _twin = self._h_twin[_curr]
            i = 0
            area = 0
            self._vertex_normals[v_idx] = 0
            self._vertex_neighbors[v_idx, :] = -1
            self._valences[v_idx] = 0

            while True:
                if (_curr == -1) or (_twin == -1):
                    break

                _vertex = self._h_vertex[_curr]
                _face = self._h_face[_curr]

                if (i < self.max_valence):
                    self._vertex_neighbors[v_idx, i] = _curr
                    n = self.face_normals[_face]
                    a = self.face_areas[_face]
                    self._vertex_normals[v_idx] += n*a
                    area += a
                l = self._vertices[_vertex] - self._vertices[self._h_vertex[self._h_prev[_curr]]]
                self._h_length[_curr] = np.sqrt((l*l).sum())
                _curr = self._h_next[_twin]
                _twin = self._h_twin[_curr]
                i += 1
                self._valences[v_idx] += 1
                if (_curr == _orig):
                    break

            if area > 0:
                self._vertex_normals[v_idx] /= area
                n = self._vertex_normals[v_idx]
                nn = np.sqrt((n*n).sum())
                if nn > 0:
                    self._vertex_normals[v_idx] /= nn
                else:
                    self._vertex_normals[v_idx] = 0
            else:
                self._vertex_normals[v_idx] = 0

    def _resize(self, vec, axis=0, skip_entries=True, return_orig=False):
        """
        Double the size of the input vector, keeping all active data.
        This is designed to expand our halfedge vectors.

        Parameters
        ---------
            vec : np.array
                Vector to expand, contains -1s in unused entries.
            axis : int
                Axis (0 or 1) along which to resize vector (only works for 2D arrays).
            skip_entries : bool
                Don't copy -1 entries to the new array.
            return_orig : bool
                Return indices of non-negative-one entries in the array prior to resize.
        """

        if (axis > 1) or (len(vec.shape) > 2):
            raise NotImplementedError('This function only works for 2D arrays along axes 0 and 1.')

        # Increase the size 1.5x along the axis.
        dt = vec.dtype
        new_size = np.array(vec.shape)
        new_size[axis] = int(1.5*new_size[axis] + 0.5)
        # new_size[axis] *= 2
        new_size = tuple(new_size)
        
        # Allocate memory for new array
        new_vec = -1*np.ones(new_size, dtype=dt)

        if skip_entries:
            # Check for invalid entries
            copy_mask = (vec != -1)
            if len(vec.shape) > 1:
                copy_mask = np.any(copy_mask, axis=(1-axis))
                copy_size = np.sum(copy_mask)
            else:
                copy_size = np.sum(copy_mask)

            if axis:
                new_vec[:,:copy_size] = vec[:, copy_mask]
            else:
                new_vec[:copy_size] = vec[copy_mask]

            # Return positions of non-negative-1 entries prior to re-ordering
            if return_orig:
                _orig = np.where(copy_mask)[0]
                # Return empty list if original values are all at the same place
                if np.all(_orig == np.arange(len(_orig))):
                    _orig = []
                return new_vec, _orig

            return new_vec

        # We're not skipping entries
        if axis:
            copy_size = vec.shape[1]
            new_vec[:,:copy_size] = vec
        else:
            copy_size = vec.shape[0]
            new_vec[:copy_size] = vec


        # Update array
        return new_vec

    def __find_idx(self, vec, axis=0):
        # Find location to insert
        _idx = np.argwhere(vec == -1)

        if len(vec.shape) > 1:
            _idx = np.argwhere(np.all(vec == -1, axis=(1-axis)))

        return _idx

    def _insert(self, vec, value, idx=None, axis=0, skip_entries=True, return_orig=False):
        """
        Insert a new value into vector vec (in place).

        Parameters
        ---------
            vec : np.array
                Vector in which to insert a new value, contains -1s in unused entries.
            value : int or np.array
                New value to insert.
            idx : int
                Specifies position of insertion in vec. Can be >len(vec).
            axis : int
                Axis (0 or 1) along which to resize vector (only works for 2D arrays).
            skip_entries : bool
                Don't copy -1 entries to the new array on resize.
            return_orig : bool
                Return indices of non-negative-one entries in the array prior to resize.

        Returns
        -------
            pos : int
                Position of value insertion.
        """

        if (axis > 1) or (len(vec.shape) > 2):
            raise NotImplementedError('This function only works for 2D arrays along axes 0 and 1.')

        if idx == None:
            # Find location to insert
            _idx = self.__find_idx(vec, axis)
        else:
            _idx = np.array([[idx]])

        # Dummy in case there's no resize
        if return_orig:
            _orig = []
        
        if (_idx.size == 0) or (_idx[0][0] > (vec.shape[axis]-1)):
            if return_orig:
                vec, _orig = self._resize(vec, axis=axis, skip_entries=skip_entries, return_orig=True)
            else:
                vec = self._resize(vec, axis=axis, skip_entries=skip_entries)
            
            if (_idx.size == 0):
                _idx = self.__find_idx(vec, axis)

        if axis:
            vec[:, _idx[0][0]] = value
        else:
            vec[_idx[0][0]] = value

        # Returns empty list for _orig if there's no resize or if
        # no positions are moved
        if return_orig:
            return vec, _idx[0][0], _orig

        return vec, _idx[0][0]

    def _insert_vertex(self, _vertex):
        """
        Insert a new vertex into the mesh.

        Parameters
        ----------
            _vertex : np.array
                1D array of x, y, z coordinates of the new vertex.
        """
        # n = self._vertices.shape[0]-1
        self._vertices, _idx, _v_orig = self._insert(self._vertices, _vertex, return_orig=True)
        self._valences, _ = self._insert(self._valences, 0)
        # if _idx > n:
        self._vertex_halfedges, _ = self._insert(self._vertex_halfedges, 0)
        self._vertex_normals, _ = self._insert(self._vertex_normals, 0)
        if self._vertex_neighbors.shape[0] < self._vertices.shape[0]:
            self._vertex_neighbors = self._resize(self._vertex_neighbors, skip_entries=False)

        if len(_v_orig) > 0:
            # Some vertices were moved, we need to update references
            # I'm not sure we'll ever make it into this case
            self._vertex_halfedges[_v_orig] = np.arange(_idx)

        self._faces_by_vertex = None  # Reset

        return _idx

    def edge_collapse(self, _curr, live_update=True):
        """
        A.k.a. delete two triangles. Remove an edge, defined by halfedge _curr, 
        and its associated triangles, while keeping mesh connectivity.

        Parameters
        ----------
            _curr: int
                Pointer to halfedge defining edge to collapse.
            live_update : bool
                Update associated faces and vertices after collapse. Set to 
                False to handle this externally (useful if operating on 
                multiple, disjoint edges).
        """

        # Create the pointers we need
        _prev = self._h_prev[_curr]
        _next = self._h_next[_curr]
        _twin = self._h_twin[_curr]
        
        # double check that we have a twin (otherwise -1 indexing will give us the last entry in the half edge list)
        # this should stop us ripping holes once we have a few -1 entries. TODO - handle more gracefully
        assert(_twin != -1)
        assert(_curr != -1)
        assert(_prev != -1)
        assert(_next != -1)
        
        _twin_prev = self._h_prev[_twin]
        _twin_next = self._h_next[_twin]
        
        assert(_twin_prev != -1)
        assert(_twin_next != -1)

        _dead_vertex = self._h_vertex[_twin]
        _live_vertex = self._h_vertex[_curr]

        # Collapse to the midpoint of the original edge vertices
        self._h_vertex[self._h_vertex == _dead_vertex] = _live_vertex
        self._vertices[_live_vertex, :] = 0.5*(self._vertices[_live_vertex, :] + self._vertices[_dead_vertex, :])
        
        # update valence of vertex we keep
        self._valences[_live_vertex] = self._valences[_live_vertex] + self._valences[_dead_vertex] -3
        
        # delete dead vertex
        self._vertices[_dead_vertex, :] = -1
        self._vertex_normals[_dead_vertex, :] = -1
        self._vertex_neighbors[_dead_vertex, :] = -1
        self._valences[_dead_vertex] = -1
        self._vertex_halfedges[_dead_vertex] = -1

        # Zipper the remaining triangles
        def _zipper(edge1, edge2):
            t1 = self._h_twin[edge1]
            t2 = self._h_twin[edge2]
            
            if (t1 != -1):
                self._h_twin[t2] = t1
                
            if (t2 != -1):
                self._h_twin[t1] = t2
                
        _zipper(_next, _prev)
        _zipper(_twin_next, _twin_prev)

        # We need some more pointers
        _prev_twin = self._h_twin[_prev]
        _prev_twin_vertex = self._h_vertex[_prev_twin]
        _next_prev_twin = self._h_next[_prev_twin]
        _next_prev_twin_vertex = self._h_vertex[_next_prev_twin]
        _twin_next_vertex = self._h_vertex[_twin_next]
        _next_twin_twin_next = self._h_next[self._h_twin[_twin_next]]
        _next_twin_twin_next_vertex = self._h_vertex[_next_twin_twin_next]

        # Make sure we have good _vertex_halfedges references
        self._vertex_halfedges[_live_vertex] = _prev_twin
        self._vertex_halfedges[_prev_twin_vertex] = _next_prev_twin
        self._vertex_halfedges[_twin_next_vertex] = self._h_twin[_twin_next]
        self._vertex_halfedges[_next_twin_twin_next_vertex] = self._h_next[_next_twin_twin_next]

        # Delete the inner triangles
        _curr_face = self._h_face[_curr]
        _twin_face = self._h_face[_twin]
        self._faces[_curr_face] = -1
        self._faces[_twin_face] = -1
        self._face_areas[_curr_face] = 0
        self._face_areas[_twin_face] = 0 
        self._face_normals[_curr_face, :] = 0
        self._face_normals[_twin_face, :] = 0

        # Delete curr, next, prev
        self._edge_delete(_curr)
        self._edge_delete(_prev)
        self._edge_delete(_next)

        # Delete _twin, _twin_prev, _twin_next
        self._edge_delete(_twin)
        self._edge_delete(_twin_prev)
        self._edge_delete(_twin_next)

        if live_update:
            # Update faces
            self._update_face_normals([self._h_face[_prev_twin], self._h_face[self._h_twin[_twin_next]]])
            self._update_vertex_neighbors([_live_vertex, _prev_twin_vertex, _next_prev_twin_vertex, _twin_next_vertex])

            self._faces_by_vertex = None

    def _edge_delete(self, _edge):
        self._h_vertex[_edge] = -1
        self._h_face[_edge] = -1
        self._h_twin[_edge] = -1
        self._h_next[_edge] = -1
        self._h_prev[_edge] = -1
        self._h_length[_edge] = -1

    def edge_split(self, _curr, live_update=True):
        """
        Split triangles evenly along an edge specified by halfedege index _curr.

        Parameters
        ----------
            _curr : int
                Pointer to halfedge defining edge to split.
            live_update : bool
                Update associated faces and vertices after split. Set to False
                to handle this externally (useful if operating on multiple, 
                disjoint edges).
        """
        _prev = self._h_prev[_curr]
        _next = self._h_next[_curr]
        _twin = self._h_twin[_curr]
        _twin_prev = self._h_prev[_twin]
        _twin_next = self._h_next[_twin]

        # Grab the new vertex position
        _vertex = 0.5*(self.vertices[self._h_vertex[_curr]] + self.vertices[self._h_vertex[_twin]])
        _vertex_idx = self._insert_vertex(_vertex)

        # Ensure the original faces have the correct pointers and add two new faces
        self._faces[self._h_face[_curr]] = _curr
        self._faces[self._h_face[_twin]] = _twin
        self._faces, _face_1_idx = self._insert(self._faces, _twin_prev)
        self._faces, _face_2_idx = self._insert(self._faces, _next)
        self._h_face[_twin_prev] = _face_1_idx
        self._h_face[_next] = _face_2_idx

        # Make sure old vertices have existing vertex halfedges
        self._vertex_halfedges[self._h_vertex[_curr]] = _next
        self._vertex_halfedges[self._h_vertex[_twin]] = _twin_next
        self._vertex_halfedges[self._h_vertex[_next]] = _prev
        self._vertex_halfedges[self._h_vertex[_twin_next]] = _twin_prev

        # Insert halfedges by face, completing the original faces first
        # Note that this assumes there's a -1 in identical locations across all halfedge vectors (should be safe)
        self._h_vertex, _he_0_idx = self._insert(self._h_vertex, self._h_vertex[_next])
        self._h_prev, _ = self._insert(self._h_prev, _curr)
        self._h_next, _ = self._insert(self._h_next, _prev)
        self._h_face, _ = self._insert(self._h_face, self._h_face[_curr])

        self._h_vertex, _he_1_idx = self._insert(self._h_vertex, _vertex_idx)
        self._h_prev, _ = self._insert(self._h_prev, _twin_next)
        self._h_next, _ = self._insert(self._h_next, _twin)
        self._h_face, _ = self._insert(self._h_face, self._h_face[_twin])

        self._h_vertex, _he_2_idx = self._insert(self._h_vertex, self._h_vertex[_twin_next])
        self._h_vertex, _he_3_idx = self._insert(self._h_vertex, _vertex_idx)
        self._h_prev, _ = self._insert(self._h_prev, _he_3_idx)
        self._h_prev, _ = self._insert(self._h_prev, _twin_prev)
        self._h_next, _ = self._insert(self._h_next, _twin_prev)
        self._h_next, _ = self._insert(self._h_next, _he_2_idx)
        self._h_face, _ = self._insert(self._h_face, _face_1_idx)
        self._h_face, _ = self._insert(self._h_face, _face_1_idx)

        self._h_vertex, _he_4_idx = self._insert(self._h_vertex, self._h_vertex[_curr])
        self._h_vertex, _he_5_idx = self._insert(self._h_vertex, _vertex_idx)
        self._h_prev, _ = self._insert(self._h_prev, _he_5_idx)
        self._h_prev, _ = self._insert(self._h_prev, _next)
        self._h_next, _ = self._insert(self._h_next, _next)
        self._h_next, _ = self._insert(self._h_next, _he_4_idx)
        self._h_face, _ = self._insert(self._h_face, _face_2_idx)
        self._h_face, _ = self._insert(self._h_face, _face_2_idx)

        # Twin updates are special because -1 is a valid twin entry
        self._h_twin, _ = self._insert(self._h_twin, _he_5_idx, idx=_he_0_idx, skip_entries=False)
        self._h_twin, _ = self._insert(self._h_twin, _he_2_idx, idx=_he_1_idx, skip_entries=False)
        self._h_twin, _ = self._insert(self._h_twin, _he_1_idx, idx=_he_2_idx, skip_entries=False)
        self._h_twin, _ = self._insert(self._h_twin, _he_4_idx, idx=_he_3_idx, skip_entries=False)
        self._h_twin, _ = self._insert(self._h_twin, _he_3_idx, idx=_he_4_idx, skip_entries=False)
        self._h_twin, _ = self._insert(self._h_twin, _he_0_idx, idx=_he_5_idx, skip_entries=False)

        # Update _prev, next
        self._h_prev[_prev] = _he_0_idx
        self._h_prev[_next] = _he_4_idx
        self._h_next[_next] = _he_5_idx

        # Update _twin_next, _twin_prev
        self._h_next[_twin_next] = _he_1_idx
        self._h_prev[_twin_prev] = _he_2_idx
        self._h_next[_twin_prev] = _he_3_idx

        # Update _curr and _twin vertex
        self._h_vertex[_curr] = _vertex_idx
        self._vertex_halfedges[_vertex_idx] = _twin
        self._h_next[_curr] = _he_0_idx
        self._h_prev[_twin] = _he_1_idx

        while len(self._face_areas) < len(self._faces):
            self._face_areas, _ = self._insert(self._face_areas, 0)
            self._face_normals, _ = self._insert(self._face_normals, 0)

        while len(self._h_length) < len(self._h_vertex):
            self._h_length, _ = self._insert(self._h_length, 0)

        # while len(self._vertex_neighbors) < len(self.vertices):
        #     self._vertex_neighbors, _ = self._insert(self._vertex_neighbors, 0)
        #     self._vertex_normals, _ = self._insert(self._vertex_normals, 0)

        if live_update:
            self._update_face_normals([self._h_face[_he_0_idx], self._h_face[_he_1_idx], self._h_face[_he_2_idx], self._h_face[_he_5_idx]])
            self._update_vertex_neighbors([self._h_vertex[_he_0_idx], self._h_vertex[_twin], self._h_vertex[_he_1_idx], self._h_vertex[_he_2_idx], self._h_vertex[_he_4_idx]])

            self._faces_by_vertex = None
    
    def edge_flip(self, _curr, live_update=True):
        """
        Flip an edge specified by halfedge index _curr.

        Parameters
        ----------
            _curr : int
                Pointer to halfedge defining edge to flip.
            live_update : bool
                Update associated faces and vertices after flip. Set to False
                to handle this externally (useful if operating on multiple, 
                disjoint edges).
        """

        _prev = self._h_prev[_curr]
        _next = self._h_next[_curr]
        _twin = self._h_twin[_curr]
        _twin_prev = self._h_prev[_twin]
        _twin_next = self._h_next[_twin]

        # Calculate adjustments to the halfedges we're flipping
        new_v0 = self._h_vertex[_next]
        new_v1 = self._h_vertex[_twin_next]

        # # _next and _twin_next are the only two halfedges switching faces
        # self._h_face[_next] = self._h_face[_twin]
        # self._h_face[_twin_next] = self._h_face[_curr]

        # _next's next and prev must be adjusted
        self._h_prev[_next] = _twin_prev
        self._h_next[_next] = _twin

        # _twin_next's next and prev must be adjusted
        self._h_prev[_twin_next] = _prev
        self._h_next[_twin_next] = _curr

        # _prev's next and prev must be updated
        self._h_prev[_prev] = _curr
        self._h_next[_prev] = _twin_next

        # _twin_prev's next and prev must be updated
        self._h_prev[_twin_prev] = _twin
        self._h_next[_twin_prev] = _next

        # Don't even bother with checking, just make sure to update the
        # vertex_halfedges references to halfedges we know work.
        self._vertex_halfedges[self._h_vertex[_curr]] = _next
        self._vertex_halfedges[self._h_vertex[_twin]] = _twin_next

        # Apply adjustments to the the halfedges we're flipping
        self._h_vertex[_curr] = new_v0
        self._h_vertex[_twin] = new_v1
        self._h_next[_curr] = _prev
        self._h_next[_twin] = _twin_prev
        self._h_prev[_curr] = _twin_next
        self._h_prev[_twin] = _next

        # Update pointers
        _prev = self._h_prev[_curr]
        _next = self._h_next[_curr]
        _twin = self._h_twin[_curr]
        _twin_prev = self._h_prev[_twin]
        _twin_next = self._h_next[_twin]

        # Set faces
        self._h_face[_next] = self._h_face[_curr]
        self._h_face[_prev] = self._h_face[_curr]
        self._h_face[_twin_next] = self._h_face[_twin]
        self._h_face[_twin_prev] = self._h_face[_twin]

        # Finish updating vertex_halfedges references, update face references.
        self._vertex_halfedges[self._h_vertex[_curr]] = _twin
        self._vertex_halfedges[self._h_vertex[_twin]] = _curr
        self._faces[self._h_face[_curr]] = _curr
        self._faces[self._h_face[_twin]] = _twin

        if live_update:
            # Update face and vertex normals
            self._update_face_normals([self._h_face[_curr], self._h_face[_twin]])
            self._update_vertex_neighbors([self._h_vertex[_curr], self._h_vertex[_twin], self._h_vertex[_next], self._h_vertex[_twin_next]])

            self._faces_by_vertex = None

    def regularize(self):
        """
        Adjust vertices so they tend toward valence < self.max_valence.
        """

        # Make sure we don't flip edges forever (we can always try a further refinement)
        n_tries = 2*self.max_valence + 1

        # n = 0
        # while True:
        # Find which vertices have high valences
        problems = np.where(self._valences > self.max_valence)[0]

            # if (problems.size == 0) or (n > n_tries):
            #     break

        # Loop over high valence vertices and flip edges to reduce valence
        for _idx in problems:
            i = 0
            while (i < n_tries) and (self._valences[_idx] > self.max_valence):
                self.edge_flip(self._vertex_halfedges[_idx], live_update=True)
                i += 1
        
            # # Now we gotta recalculate the normals
            # self._face_normals = None
            # self._vertex_normals = None
            # self.face_normals
            # self.vertex_normals

            # self._faces_by_vertex = None
            # n += 1

    def relax(self, l=1, n=1):
        """
        Perform n iterations of Lloyd relaxation on the mesh.

        Parameters
        ----------
            l : float
                Regularization (damping) term, used to avoid oscillations.
            n : int
                Number of iterations to apply.
        """

        for k in range(n):
            # Get vertex neighbors
            nn = self._vertex_neighbors
            nn_mask = (self._vertex_neighbors != -1)
            vn = self._vertices[self._h_vertex[nn]]
            
            # Get the faces of the vertex neighbors
            # fn = self._h_face[nn]

            # Get the face centroids
            # fc = 0.33*(self._vertices[self._h_vertex[self._h_prev[nn]]] + self._vertices[self._h_vertex[nn]] + self._vertices[self._h_vertex[self._h_next[nn]]])

            # an = self._face_areas[fn]*nn_mask
            # Calculate the voronoi areas and mask off the wrong ones
            # an = 0.5*np.linalg.norm(np.cross(np.diff(fc, axis=1), (self._vertices[:,None,:]-fc)[:,:(self.max_valence-1),:], axis=2), axis=2)*nn_mask
            an = (1./self._h_length[nn])*nn_mask
            an[self._h_length[nn] == 0] = 0

            # Calculate gravity-weighted centroids
            A = np.sum(an, axis=1)
            c = 1./(A[...,None])*np.sum(an[...,None]*vn, axis=1)
            # Don't get messed up by slivers. TODO: This is a hack. There should be
            # sliver elimination.
            c[A == 0] = self._vertices[A == 0]

            # Construct projection vector into tangent plane
            pn = self._vertex_normals[...,None]*self._vertex_normals[:,None,:]
            p = np.eye(3)[None,:] - pn

            # Update vertex positions
            self._vertices = self._vertices + l*np.sum(p*((c-self._vertices)[...,None]),axis=1)

        # Now we gotta recalculate the normals
        self._face_normals = None
        self._vertex_normals = None
        self.face_normals
        self.vertex_normals

    def remesh(self, edge_length=-1, l=1, n=1, n_relax=1):
        """
        Produce a higher-quality mesh.

        Follows procedure in Botsch and Kobbelt, A Remeshing Approach 
        to Multiresoluton Modeling, Eurographics Symposium on Geometry 
        Processing, 2004.

        Parameters
        ----------
            edge_length : float
                Target edge length for all edges in the mesh.
            l : float
                Relaxation regularization term, used to avoid oscillations.
            n : int
                Number of remeshing iterations to apply.
            n_relax : int 
                Number of Lloyd relaxation (relax()) iterations to apply 
                per remeshing iteration.
        """

        if (edge_length == -1):
            # Guess edge_length
            edge_length = np.mean(self._h_length[self._h_length != -1])

        for k in range(n):
            # # 1. Split all edges longer than (4/3)*edge_length at their midpoint.
            # split_idxs = np.where((self._h_length > 1.33*edge_length)*(self._h_length != -1))[0]
            # d = {}
            # for i in split_idxs:
            #     _twin = self._h_twin[i]

            #     if _twin in d.keys():
            #         continue

            #     d[i] = _twin
            #     self.edge_split(i)
            
            # # 2. Collapse all edges shorter than (4/5)*edge_length to their midpoint.
            # collapse_idxs = np.where((self._h_length < 0.8*edge_length)*(self._h_length != -1))[0]
            # d = {}
            # for i in collapse_idxs:
            #     _twin = self._h_twin[i]

            #     if (_twin == -1) or (_twin in d.keys()):
            #         continue

            #     d[i] = _twin
            #     self.edge_collapse(i)

            collapse_count = 0
            for i in np.arange(len(self._h_length)):
                if (self._h_length[i] != -1) and (self._h_length[i] < 0.8*edge_length):
                    self.edge_collapse(i)
                    collapse_count += 1
            print('Collapse count: ' + str(collapse_count))

            # # 3. Flip edges in order to minimize deviation from valence 6.
            # self.regularize()

            # # 4. Relocate vertices on the surface by tangential smoothing.
            # self.relax(l=l, n=n_relax)

    def repair(self):
        """
        Make the mesh manifold.
        """
        pass

        # # Unordered halfedges
        # edges = np.vstack([self.faces[:,[0,1]], self.faces[:,[1,2]], self.faces[:,[2,0]]])

        # # Group edges uniquely
        # packed_edges = pack_edges(edges)

        # # Return the unique edges and their counts
        # e, c = np.unique(packed_edges, return_counts=True)

        # # Get the bad edges
        # bad_edges = dict([(x,i) for i, x in enumerate(e[c > 2])])
        # # print(bad_edges)

        # n = len(self.faces)

        # # Delete all edges of degree > 2
        # for i, edge in enumerate(packed_edges):
        #     # print(edge)
        #     if not bool(bad_edges):
        #         break
        #     if edge in bad_edges.keys():
        #         bad_edges.pop(edge)
        #         # Delete the edge
        #         _idx = np.mod(i, n)
        #         self._h_vertex[_idx] = -1
        #         self._h_twin[_idx] = -1
        #         self._h_prev[_idx] = -1
        #         self._h_next[_idx] = -1
        #         self._h_face[_idx] = -1
        #         self._h_length[_idx] = -1

        # While there are vertices connected to two -1 halfedges
        # while True:
        #     # Find vertices sharing two -1 halfedges
        #     no_twin = (self._h_twin == -1)
        #     _he1, _he2 = np.where(self._h_vertex[no_twin][:,None] == self._h_vertex[self._h_prev[no_twin]][None,:])

        #     if _he1.size == 0:
        #         break

        #     # Create triangles from these vertices
        #     for _curr in orphans:
        #         _twin = self._h_twin[_curr]
        #         _twin_prev = self._h_prev[_twin]
        #         _twin_next = self._h_next[_twin]
        #         self._h_vertex, _he_0_idx = self._insert(self._h_vertex, self._h_vertex[_next])
        #         self._h_prev, _ = self._insert(self._h_prev, _curr)
        #         self._h_next, _ = self._insert(self._h_next, _prev)
        #         self._h_face, _ = self._insert(self._h_face, self._h_face[_curr])

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

        dt = np.dtype([('normal', '3f4'), ('vertex0', '3f4'), ('vertex1', '3f4'), 
                       ('vertex2', '3f4'), ('attrib', 'u2')])

        triangles_stl = np.zeros(self.face_normals.shape[0], dtype=dt)
        triangles_stl['vertex0'] = self.vertices[self.faces[:, 0]]
        triangles_stl['vertex1'] = self.vertices[self.faces[:, 1]]
        triangles_stl['vertex2'] = self.vertices[self.faces[:, 2]]
        triangles_stl['normal'] = self.face_normals

        stl.save_stl_binary(filename, triangles_stl)