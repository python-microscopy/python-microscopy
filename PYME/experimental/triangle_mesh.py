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

HALFEDGE_DTYPE = np.dtype([('vertex', 'i4'), ('face', 'i4'), ('twin', 'i4'), ('next', 'i4'), ('prev', 'i4'), ('length', 'f4')])
FACE_DTYPE = np.dtype([('halfedge', 'i4'), ('normal', '3f4'), ('area', 'f4')])
VERTEX_DTYPE = np.dtype([('position', '3f4'), ('normal', '3f4'), ('halfedge', 'i4'), ('valence', 'i4'), ('neighbors', '6i4')])

class TriangleMesh(object):
    def __init__(self, vertices, faces, debug=False):
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
            debug : bool
                Print debug statements (assumes manifold mesh).
        """
        self._vertices = np.zeros(vertices.shape[0], dtype=VERTEX_DTYPE)
        self._vertices[:] = -1  # initialize everything to -1 to start with
        self._vertices['position'] = vertices
        self._vertex_vacancies = []

        self._faces = None  # Contains a pointer to one halfedge associated with each face
        self._faces_by_vertex = None  # Representation of faces by triplets of vertices
        self._face_vacancies = []

        # Halfedges
        self._halfedges = None
        self._halfedge_vacancies = []
        self._initialize_halfedges(vertices, faces)

        # Populate the normals
        self.face_normals
        self.vertex_normals

        # Properties we can visualize
        self.vertex_properties = ['x', 'y', 'z']

        # Print debug statements (assumes manifold mesh)
        self.debug = debug

    def __getitem__(self, k):
        # this defers evaluation of the properties until we actually access them, 
        # as opposed to the mappings which stored the values on class creation.
        try:
            res = getattr(self, k)
        except AttributeError:
            raise KeyError('Key %s not defined' % k)
        
        return res

    @classmethod
    def from_stl(cls, filename, debug=False):
        """
        Read from an STL file.
        """
        from PYME.IO.FileUtils import stl

        # Load an STL from file
        triangles_stl = stl.load_stl_binary(filename)

        # Call from_np_stl on the file stream
        return cls.from_np_stl(triangles_stl, debug=debug)

    @classmethod
    def from_np_stl(cls, triangles_stl, debug=False):
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

        return cls(vertices, faces, debug=debug)

    @property
    def _h_vertex(self):
        return self._halfedges['vertex']

    @property
    def _h_face(self):
        return self._halfedges['face']

    @property
    def _h_twin(self):
        return self._halfedges['twin']

    @property
    def _h_next(self):
        return self._halfedges['next']

    @property
    def _h_prev(self):
        return self._halfedges['prev']

    @property
    def _h_length(self):
        return self._halfedges['length']

    @property
    def _valences(self):
        return self._vertices['valence']
    
    @property
    def _vertex_neighbors(self):
        return self._vertices['neighbors']

    @property
    def _vertex_normals(self):
        return self._vertices['normal']

    @property
    def _vertex_halfedges(self):
        return self._vertices['halfedge']

    @property
    def _face_normals(self):
        return self._faces['normal']

    @property
    def _face_areas(self):
        return self._faces['area']

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
        return self._vertices['position']

    @property
    def faces(self):
        """
        Returns faces by vertex for rendering and STL applications.
        """
        if self._faces_by_vertex is None:
            faces = self._faces['halfedge'][self._faces['halfedge'] != -1]
            v0 = self._halfedges['vertex'][self._halfedges['prev'][faces]]
            v1 = self._halfedges['vertex'][faces]
            v2 = self._halfedges['vertex'][self._halfedges['next'][faces]]
            self._faces_by_vertex = np.vstack([v0, v1, v2]).T
        return self._faces_by_vertex

    @property
    def face_normals(self):
        """
        Calculate and return the normal of each triangular face.
        """
        if np.all(self._faces['normal'] == -1):
            v2 = self.vertices[self._halfedges['vertex'][self._halfedges['prev'][self._faces['halfedge']]]]
            v1 = self.vertices[self._halfedges['vertex'][self._faces['halfedge']]]
            v0 = self.vertices[self._halfedges['vertex'][self._halfedges['next'][self._faces['halfedge']]]]
            u = v2 - v1
            v = v0 - v1
            n = np.cross(u, v, axis=1)
            nn = np.linalg.norm(n, axis=1)
            self._faces['area'] = 0.5*nn
            self._faces['normal'] = n/nn[:, None]
            self._faces['normal'][np.isnan(self._faces['normal'])] = 0
            self._faces['normal'][self._faces['halfedge'] == -1] = 0
        return self._faces['normal']

    @property
    def face_areas(self):
        """
        Calculate and return the area of each triangular face.
        """
        if np.all(self._faces['area'] == -1):
            self._faces['normal'][:] = -1
            self.face_normals
        return self._faces['area']
    
    @property
    def vertex_neighbors(self):
        """
        Return the up to self.max_valence neighbors of each vertex.
        """
        if np.all(self._vertices['neighbors'] == -1):
            for v_idx in np.arange(len(self.vertices)):
                _orig = self._vertices['halfedge'][v_idx]
                _curr = _orig
                _twin = self._halfedges['twin'][_curr]
                i = 0
                while True:
                    if (_curr == -1) or (_twin == -1):
                        break
                    _vertex = self._halfedges['vertex'][_curr]
                    _face = self._halfedges['face'][_curr]
                    if (i < 6):
                        self._vertices['neighbors'][v_idx, i] = _curr
                        n = self.face_normals[_face]
                        a = self.face_areas[_face]
                        self._vertices['normal'][v_idx] += n*a
                    l = self._vertices['position'][_vertex] - self._vertices['position'][self._halfedges['vertex'][self._halfedges['prev'][_curr]]]
                    self._halfedges['length'][_curr] = np.sqrt((l*l).sum())
                    _curr = self._halfedges['next'][_twin]
                    _twin = self._halfedges['twin'][_curr]
                    i += 1
                    if (_curr == _orig):
                        break
                self._vertices['valence'][v_idx] = i
            nn = np.linalg.norm(self._vertices['normal'], axis=1)
            self._vertices['normal'] /= nn[:, None]
            self._vertices['normal'][np.isnan(self._vertices['normal'])] = 0

        return self._vertices['neighbors']

    @property
    def vertex_normals(self):
        """
        Calculate and return the normal of each vertex.
        """
        if np.all(self._vertices['normal'] == -1):
            self._vertices['neighbors'][:] = -1
            self.vertex_neighbors
        return self._vertices['normal']

    @property
    def valences(self):
        """
        Calculate and return the valence of each vertex.
        """
        if np.all(self._vertices['valence'] == -1):
            self._vertices['neighbors'][:] = -1
            self.vertex_neighbors
        return self._vertices['valence']

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
            n_edges = len(edges)
            j = np.arange(n_faces)
            
            self._halfedges = np.zeros(n_edges, dtype=HALFEDGE_DTYPE)
            self._halfedges[:] = -1  # initialize everything to -1 to start with

            self._halfedges['vertex'] = edges[:, 1]
            
            self._faces = np.zeros(n_faces, dtype=FACE_DTYPE)
            self._faces[:] = -1  # initialize everything to -1 to start with

            # Create a unique handle for each edge
            edges_packed = pack_edges(edges)
            
            # Use a dictionary to keep track of which edges are already assigned twins
            d = {}
            for i, e in enumerate(edges_packed):
                if e in list(d.keys()):
                    idx = d.pop(e)
                    self._halfedges['twin'][idx] = i
                    self._halfedges['twin'][i] = idx
                else:
                    d[e] = i

            self._halfedges['face'] = np.hstack([j, j, j])
            self._halfedges['next'] = np.hstack([j+n_faces, j+2*n_faces, j])
            self._halfedges['prev'] = np.hstack([j+2*n_faces, j, j+n_faces])
            self._halfedges['length'] = np.linalg.norm(self._vertices['position'][self._halfedges['vertex']] - self._vertices['position'][self._halfedges['vertex'][self._halfedges['prev']]], axis=1)
            
            # Set the vertex halfedges (multiassignment, but should be fine)
            self._vertices['halfedge'][self._halfedges['vertex']] = self._halfedges['next']

            self._faces['halfedge'] = j  # Faces are defined by associated halfedges
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

        for f_idx in f_idxs:
            v2 = self.vertices[self._halfedges['vertex'][self._halfedges['prev'][self._faces['halfedge'][f_idx]]]]
            v1 = self.vertices[self._halfedges['vertex'][self._faces['halfedge'][f_idx]]]
            v0 = self.vertices[self._halfedges['vertex'][self._halfedges['next'][self._faces['halfedge'][f_idx]]]]
            u = v2 - v1
            v = v0 - v1
            n = fast_3x3_cross(u.squeeze(), v.squeeze())
            nn = np.sqrt((n*n).sum())
            self._faces['area'][f_idx] = 0.5*nn
            if nn > 0:
                self._faces['normal'][f_idx] = n/nn
            else:
                self._faces['normal'][f_idx] = 0

    def _update_vertex_neighbors(self, v_idxs):
        """
        Recalculate len(v_idxs) vertex neighbors/normals.

        Parameters
        ----------
            v_idxs : list or np.array
                List of vertex indicies indicating which vertices to update.
        """
        
        for v_idx in v_idxs:
            _orig = self._vertices['halfedge'][v_idx]
            _curr = _orig
            
            curr_edge = self._halfedges[_curr]
            _twin = curr_edge['twin']
            twin_edge = self._halfedges[_twin]
            
            if self.debug and (twin_edge['vertex'] != v_idx):
                print(_curr, curr_edge, _twin,twin_edge)
                raise RuntimeError('Twin (%d) should point back to starting vertex (%d) but points to %d' % (_twin, v_idx, twin_edge['vertex']))
            
            i = 0
            self._vertices['neighbors'][v_idx] = -1
            
            _normal = 0*self._vertices['normal'][v_idx]  # avoid a few lookups by using a local variable

            while True:
                if (_curr == -1) or (_twin == -1):
                    break

                if (i < 6):
                    self._vertices['neighbors'][v_idx, i] = _curr
                    
                    #TODO - should these be in if clause?
                    _face = curr_edge['face']
                    n = self._faces['normal'][_face]
                    a = self._faces['area'][_face]
                    _normal += n*a
                else:
                    if self.debug and (i > 20):
                        raise RuntimeError('Abnormal vertex valance detected on vertex %d' % v_idx)
                
                vertex = self._vertices['position'][v_idx]
                l = vertex - self._vertices['position'][curr_edge['vertex']]
                l = np.sqrt((l*l).sum())
                curr_edge['length'] = l
                twin_edge['length'] = l
                
                _curr = twin_edge['next']
                curr_edge = self._halfedges[_curr]
                _twin = curr_edge['twin']
                twin_edge = self._halfedges[_twin]
                
                i += 1
                
                if (_curr == _orig):
                    break

            self._vertices['valence'][v_idx] = i

            nn = np.sqrt((_normal*_normal).sum())
            if nn > 0:
                self._vertices['normal'][v_idx] = _normal/nn
            else:
                self._vertices['normal'][v_idx] = 0

    def _resize(self, vec, axis=0, skip_entries=True, return_orig=False, key=None):
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
            key : string
                Optional array key on which to check for -1 value.
        """

        if (axis > 1) or (len(vec.shape) > 2):
            raise NotImplementedError('This function only works for 2D arrays along axes 0 and 1.')

        # Increase the size 1.5x along the axis.
        dt = vec.dtype
        new_size = np.array(vec.shape)
        new_size[axis] = int(1.5*new_size[axis] + 0.5)
        new_size = tuple(new_size)
        
        # Allocate memory for new array
        new_vec = np.empty(new_size, dtype=dt)
        new_vec[:] = -1

        if skip_entries:
            # Check for invalid entries
            if key:
                copy_mask = (vec[key] != 1)
            else:
                copy_mask = (vec != -1)
            
            if len(copy_mask.shape) > 1:
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
        else:
            # We're not skipping entries
            if axis:
                copy_size = vec.shape[1]
                new_vec[:,:copy_size] = vec
            else:
                copy_size = vec.shape[0]
                new_vec[:copy_size] = vec
                
            if return_orig:
                return new_vec, []
    
            # Update array
            return new_vec

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
        curr_halfedge = self._halfedges[_curr]
        _prev = curr_halfedge['prev']
        _next = curr_halfedge['next']
        _twin = curr_halfedge['twin']
        
        # double check that we have a twin (otherwise -1 indexing will give us the 
        # last entry in the half edge list) this should stop us ripping holes once 
        # we have a few -1 entries. TODO - handle more gracefully
        if self.debug:
            assert(_twin != -1)
            assert(_curr != -1)
            assert(_prev != -1)
            assert(_next != -1)
        
        twin_halfedge = self._halfedges[_twin]
        _twin_prev = twin_halfedge['prev']
        _twin_next = twin_halfedge['next']
        
        if self.debug:
            assert(_twin_prev != -1)
            assert(_twin_next != -1)

        _dead_vertex = twin_halfedge['vertex']
        _live_vertex = curr_halfedge['vertex']
        
        if self.debug and (_live_vertex == _dead_vertex):
            print(_curr, curr_halfedge, _twin, twin_halfedge)
            raise RuntimeError('Live vertex equals dead vertex (both halves of edge point to same place)')

        # Collapse to the midpoint of the original edge vertices
        self._halfedges['vertex'][self._halfedges['vertex'] == _dead_vertex] = _live_vertex
        self._vertices['position'][_live_vertex, :] = 0.5*(self.vertices[_live_vertex, :] + self.vertices[_dead_vertex, :])
        
        # update valence of vertex we keep
        self._vertices['valence'][_live_vertex] = self._vertices['valence'][_live_vertex] + self._vertices['valence'][_dead_vertex] - 3
        
        if self.debug:
            print(self._vertices['valence'][_live_vertex], self._vertices['valence'][_dead_vertex])
            assert(self._vertices['valence'][_live_vertex] >=3)
        
        # delete dead vertex
        self._vertices[_dead_vertex] = -1

        # Zipper the remaining triangles
        def _zipper(edge1, edge2):
            t1 = self._halfedges[edge1]['twin']
            t2 = self._halfedges[edge2]['twin']
            
            if (t2 != -1):
                self._halfedges[t2]['twin'] = t1
                
            if (t1 != -1):
                self._halfedges[t1]['twin'] = t2
                
        _zipper(_next, _prev)
        _zipper(_twin_next, _twin_prev)

        # We need some more pointers
        _prev_twin = self._halfedges[_prev]['twin']
        _prev_twin_vertex = self._halfedges[_prev_twin]['vertex']
        _next_prev_twin = self._halfedges[_prev_twin]['next']
        _next_prev_twin_vertex = self._halfedges[_next_prev_twin]['vertex']
        _twin_next_vertex = self._halfedges[_twin_next]['vertex']
        _next_twin_twin_next = self._halfedges['next'][self._halfedges[_twin_next]['twin']]
        _next_twin_twin_next_vertex = self._halfedges[_next_twin_twin_next]['vertex']

        # Make sure we have good _vertex_halfedges references
        self._vertices['halfedge'][_live_vertex] = _prev_twin
        self._vertices['halfedge'][_prev_twin_vertex] = _next_prev_twin
        self._vertices['halfedge'][_twin_next_vertex] = self._halfedges[_twin_next]['twin']
        self._vertices['halfedge'][_next_twin_twin_next_vertex] = self._halfedges['next'][_next_twin_twin_next]

        # Delete the inner triangles
        _curr_face = curr_halfedge['face']
        _twin_face = twin_halfedge['face']
        self._faces['halfedge'][_curr_face] = -1
        self._faces['halfedge'][_twin_face] = -1
        self._faces['area'][_curr_face] = 0
        self._faces['area'][_twin_face] = 0 
        self._faces['normal'][_curr_face, :] = 0
        self._faces['normal'][_twin_face, :] = 0

        if self.debug:
            print(curr_halfedge, twin_halfedge)

        # Delete curr, next, prev
        self._edge_delete(_curr)
        self._edge_delete(_prev)
        self._edge_delete(_next)

        # Delete _twin, _twin_prev, _twin_next
        self._edge_delete(_twin)
        self._edge_delete(_twin_prev)
        self._edge_delete(_twin_next)

        try:
            if live_update:
                # Update faces
                self._update_face_normals([self._halfedges[_prev_twin]['face'], self._halfedges[self._halfedges[_twin_next]['twin']]['face']])
                self._update_vertex_neighbors([_live_vertex, _prev_twin_vertex, _next_prev_twin_vertex, _twin_next_vertex])
    
                self._faces_by_vertex = None
            
        except RuntimeError as e:
            print (_curr, _twin, _next, _prev, _twin_next, _twin_prev, _next_prev_twin, _next_twin_twin_next, _prev_twin)
            print([_live_vertex, _prev_twin_vertex, _next_prev_twin_vertex, _twin_next_vertex])
            raise e

    def _edge_delete(self, _edge):
        """
        Delete edge _edge.

        Parameters
        ----------
            edge : int
                One self._halfedge index of the edge to delete.
        """
        self._halfedges[_edge] = -1
        self._halfedge_vacancies.append(_edge)

    def _insert(self, el, el_arr, el_vacancies_arr, key, compact=False, **kwargs):
        try:
            idx = el_vacancies_arr.pop(0)
        except IndexError:
            # no vacant slot, resize
            el_arr = self._resize(el_arr, skip_entries=compact, key=key)
            #TODO - invalidate neighbours, vertex_halfedges etc ???
            
            if len(el_arr[key].shape) > 1:
                el_vacancies_arr = [int(x) for x in np.argwhere(np.sum(el_arr[key],axis=1)/el_arr[key].shape[1] == -1)]
            else:
                el_vacancies_arr = [int(x) for x in np.argwhere(el_arr[key] == -1)]

            idx = el_vacancies_arr.pop(0)
            
        ed = el_arr[idx]
        ed[key] = el
        for k, v in kwargs.items():
            ed[k] = v
            
        return ed, idx, el_arr, el_vacancies_arr

    def _new_edge(self, vertex, compact=False, **kwargs):
        """
        Create a new edge.

        Parameters
        ----------
            vertex : idx
                self._vertices index which this halfedge points to
            compact : bool
                Do we copy -1 values in the resize of _halfedges?
            kwargs 
                List of _halfedge dtype parameters (e.g. prev=10, next=2) to define halfedge
                via more than just its vertex.

        Returns
        -------
            ed : HALFEDGE_DTYPE
                Halfedge created
            idx : int
                Index of the halfedge in self._halfedges.
        """

        ed, idx, self._halfedges, self._halfedge_vacancies = self._insert(vertex, self._halfedges, self._halfedge_vacancies, 'vertex', compact, **kwargs)
            
        return ed, idx

    def _new_face(self, _edge, compact=False, **kwargs):
        """
        Create a new face based on edge _edge.
        """

        fa, idx, self._faces, self._face_vacancies = self._insert(_edge, self._faces, self._face_vacancies, 'halfedge', compact, **kwargs)

        return fa, idx

    def _new_vertex(self, _vertex, compact=False, **kwargs):
        """
        Insert a new vertex into the mesh.

        Parameters
        ----------
            _vertex : np.array
                1D array of x, y, z coordinates of the new vertex.
        """
        vx, idx, self._vertices, self._vertex_vacancies = self._insert(_vertex, self._vertices, self._vertex_vacancies, 'position', compact, **kwargs)

        self._faces_by_vertex = None  # Reset

        return vx, idx

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
        curr_edge = self._halfedges[_curr]
        _prev = curr_edge['prev']
        _next = curr_edge['next']
        _twin = curr_edge['twin']
        
        twin_edge = self._halfedges[_twin]
        _twin_prev = twin_edge['prev']
        _twin_next = twin_edge['next']
        
        if self.debug:
            assert(_curr != -1)
            assert(_prev != -1)
            assert(_next != -1)
            assert(_twin != -1)
            assert(_twin_prev != -1)
            assert(_twin_next != -1)

        # Grab the new vertex position
        _vertex = 0.5*(self.vertices[curr_edge['vertex']] + self.vertices[twin_edge['vertex']])
        _, _vertex_idx = self._new_vertex(_vertex)
        
        # Ensure the original faces have the correct pointers and add two new faces
        self._faces[curr_edge['face']] = _curr
        self._faces[twin_edge['face']] = _twin
        _, _face_1_idx = self._new_face(_twin_prev)
        _, _face_2_idx = self._new_face(_next)
        self._halfedges[_twin_prev]['face'] = _face_1_idx
        self._halfedges[_next]['face'] = _face_2_idx

        # Insert the new faces
        _, _he_0_idx = self._new_edge(self._halfedges[_next]['vertex'], prev=_curr, next=_prev, face=curr_edge['face'])
        _, _he_1_idx = self._new_edge(_vertex_idx, prev=_twin_next, next=_twin, face=twin_edge['face'])
        
        _, _he_2_idx = self._new_edge(self._halfedges[_twin_next]['vertex'], next=_twin_prev, face=_face_1_idx)
        _, _he_3_idx = self._new_edge(_vertex_idx, prev=_twin_prev, next=_he_2_idx, face=_face_1_idx)
        self._halfedges[_he_2_idx]['prev'] = _he_3_idx

        _, _he_4_idx = self._new_edge(curr_edge['vertex'], next=_next, face=_face_2_idx)
        _, _he_5_idx = self._new_edge(_vertex_idx, prev=_next, next=_he_4_idx, face=_face_2_idx)
        self._halfedges[_he_4_idx]['prev'] = _he_5_idx

        self._halfedges[_he_0_idx]['twin'] = _he_5_idx
        self._halfedges[_he_5_idx]['twin'] = _he_0_idx

        self._halfedges[_he_1_idx]['twin'] = _he_2_idx
        self._halfedges[_he_2_idx]['twin'] = _he_1_idx

        self._halfedges[_he_3_idx]['twin'] = _he_4_idx
        self._halfedges[_he_4_idx]['twin'] = _he_3_idx

        # Make sure old vertices have existing vertex halfedges
        self._vertices['halfedge'][curr_edge['vertex']] = _he_3_idx
        self._vertices['halfedge'][twin_edge['vertex']] = _curr
        self._vertices['halfedge'][self._halfedges[_next]['vertex']] = _he_5_idx
        self._vertices['halfedge'][self._halfedges[_twin_next]['vertex']] = _he_1_idx

        # Update _prev, next
        self._halfedges[_prev]['prev'] = _he_0_idx
        self._halfedges[_next]['prev'] = _he_4_idx
        self._halfedges[_next]['next'] = _he_5_idx

        # Update _twin_next, _twin_prev
        self._halfedges[_twin_next]['next'] = _he_1_idx
        self._halfedges[_twin_prev]['prev'] = _he_2_idx
        self._halfedges[_twin_prev]['next'] = _he_3_idx

        # Update _curr and _twin vertex
        self._halfedges[_curr]['vertex'] = _vertex_idx
        self._vertices['halfedge'][_vertex_idx] = _twin
        self._halfedges[_curr]['next'] = _he_0_idx
        self._halfedges[_twin]['prev'] = _he_1_idx

        if live_update:
            self._update_face_normals([self._halfedges['face'][_he_0_idx], self._halfedges['face'][_he_1_idx], self._halfedges['face'][_he_2_idx], self._halfedges['face'][_he_5_idx]])
            self._update_vertex_neighbors([self._halfedges['vertex'][_he_0_idx], self._halfedges['vertex'][_twin], self._halfedges['vertex'][_he_1_idx], self._halfedges['vertex'][_he_2_idx], self._halfedges['vertex'][_he_4_idx]])

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

        _prev = self._halfedges['prev'][_curr]
        _next = self._halfedges['next'][_curr]
        _twin = self._halfedges['twin'][_curr]
        _twin_prev = self._halfedges['prev'][_twin]
        _twin_next = self._halfedges['next'][_twin]

        # Calculate adjustments to the halfedges we're flipping
        new_v0 = self._halfedges['vertex'][_next]
        new_v1 = self._halfedges['vertex'][_twin_next]

        # _next's next and prev must be adjusted
        self._halfedges['prev'][_next] = _twin_prev
        self._halfedges['next'][_next] = _twin

        # _twin_next's next and prev must be adjusted
        self._halfedges['prev'][_twin_next] = _prev
        self._halfedges['next'][_twin_next] = _curr

        # _prev's next and prev must be updated
        self._halfedges['prev'][_prev] = _curr
        self._halfedges['next'][_prev] = _twin_next

        # _twin_prev's next and prev must be updated
        self._halfedges['prev'][_twin_prev] = _twin
        self._halfedges['next'][_twin_prev] = _next

        # Don't even bother with checking, just make sure to update the
        # vertex_halfedges references to halfedges we know work.
        self._vertices['halfedge'][self._halfedges['vertex'][_curr]] = _next
        self._vertices['halfedge'][self._halfedges['vertex'][_twin]] = _twin_next

        # Apply adjustments to the the halfedges we're flipping
        self._halfedges['vertex'][_curr] = new_v0
        self._halfedges['vertex'][_twin] = new_v1
        self._halfedges['next'][_curr] = _prev
        self._halfedges['next'][_twin] = _twin_prev
        self._halfedges['prev'][_curr] = _twin_next
        self._halfedges['prev'][_twin] = _next

        # Update pointers
        _prev = self._halfedges['prev'][_curr]
        _next = self._halfedges['next'][_curr]
        _twin = self._halfedges['twin'][_curr]
        _twin_prev = self._halfedges['prev'][_twin]
        _twin_next = self._halfedges['next'][_twin]

        # Set faces
        self._halfedges['face'][_next] = self._halfedges['face'][_curr]
        self._halfedges['face'][_prev] = self._halfedges['face'][_curr]
        self._halfedges['face'][_twin_next] = self._halfedges['face'][_twin]
        self._halfedges['face'][_twin_prev] = self._halfedges['face'][_twin]

        # Finish updating vertex_halfedges references, update face references.
        self._vertices['halfedge'][self._halfedges['vertex'][_curr]] = _twin
        self._vertices['halfedge'][self._halfedges['vertex'][_twin]] = _curr
        self._faces[self._halfedges['face'][_curr]] = _curr
        self._faces[self._halfedges['face'][_twin]] = _twin

        if live_update:
            # Update face and vertex normals
            self._update_face_normals([self._halfedges['face'][_curr], self._halfedges['face'][_twin]])
            self._update_vertex_neighbors([self._halfedges['vertex'][_curr], self._halfedges['vertex'][_twin], self._halfedges['vertex'][_next], self._halfedges['vertex'][_twin_next]])

            self._faces_by_vertex = None

    def regularize(self):
        """
        Adjust vertices so they tend toward valence < self.max_valence.
        """

        # Make sure we don't flip edges forever (we can always try a further refinement)
        # problems = np.where((self._vertices['valence'] != 6) & (self._vertices['valence'] != -1))[0]

        # Loop over high valence vertices and flip edges to reduce valence
        for _idx, val in enumerate(self._vertices['valence']):
            if (val == -1) or (val == 6):
                continue
            i = 0
            tries = np.abs(val - 6)
            while (i < tries) and (self._vertices['valence'][_idx] != 6):
                self.edge_flip(self._vertices['halfedge'][_idx])
                i += 1

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
            nn = self._vertices['neighbors']
            nn_mask = (self._vertices['neighbors'] != -1)
            vn = self._vertices['position'][self._halfedges['vertex'][nn]]
            
            # Get the faces of the vertex neighbors
            # fn = self._halfedges['face'][nn]

            # Get the face centroids
            # fc = 0.33*(self._vertices[self._halfedges['vertex'][self._halfedges['prev'][nn]]] + self._vertices[self._halfedges['vertex'][nn]] + self._vertices[self._halfedges['vertex'][self._halfedges['next'][nn]]])

            # an = self._face_areas[fn]*nn_mask
            # Calculate the voronoi areas and mask off the wrong ones
            # an = 0.5*np.linalg.norm(np.cross(np.diff(fc, axis=1), (self._vertices[:,None,:]-fc)[:,:(self.max_valence-1),:], axis=2), axis=2)*nn_mask
            an = (1./self._halfedges['length'][nn])*nn_mask
            an[self._halfedges['length'][nn] == 0] = 0

            # Calculate gravity-weighted centroids
            A = np.sum(an, axis=1)
            c = 1./(A[...,None])*np.sum(an[...,None]*vn, axis=1)
            # Don't get messed up by slivers. TODO: This is a hack. There should be
            # sliver elimination.
            c[A == 0] = self._vertices['position'][A == 0]

            # Construct projection vector into tangent plane
            pn = self._vertices['normal'][...,None]*self._vertices['normal'][:,None,:]
            p = np.eye(3)[None,:] - pn

            # Update vertex positions
            self._vertices['position'] = self._vertices['position'] + l*np.sum(p*((c-self._vertices['position'])[...,None]),axis=1)

        # Now we gotta recalculate the normals
        self._faces['normal'][:] = -1
        self._vertices['normal'][:] = -1
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
            edge_length = np.mean(self._halfedges['length'][self._halfedges['length'] != -1])

        for k in range(n):
            # 1. Split all edges longer than (4/3)*edge_length at their midpoint.
            split_count = 0
            for i in np.arange(len(self._halfedges['length'])):
                if (self._halfedges['length'][i] >= 0) and (self._halfedges['length'][i] > 1.33*edge_length):
                    self.edge_split(i)
                    split_count += 1
            print('Split count: %d' % (split_count))
            
            if self.debug:
                #check for dud half edges
                _valid = self._halfedges['vertex'] != -1
                assert(not np.any(self._halfedges[_valid]['twin'] == -1))
                assert (not np.any(self._halfedges[_valid]['next'] == -1))
                assert (not np.any(self._halfedges[self._halfedges[_valid]['next']]['vertex'] == -1))

                assert (not np.any(self._halfedges[self._halfedges[_valid]['twin']]['next'] == self._halfedges[self._halfedges[_valid]['prev']]['twin']))
                
                assert(self._vertices['valence'][self._vertices['valence']>0].min() >=3)
        
            
            # 2. Collapse all edges shorter than (4/5)*edge_length to their midpoint.
            collapse_count = 0
            for i in np.arange(len(self._halfedges['length'])):
                if (self._halfedges['length'][i] >=0) and (self._halfedges['length'][i] < 0.8*edge_length):
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