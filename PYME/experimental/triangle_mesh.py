import numpy as np

from PYME.experimental import triangle_mesh_utils

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

HALFEDGE_DTYPE = np.dtype([('vertex', 'i4'), ('face', 'i4'), ('twin', 'i4'), ('next', 'i4'), ('prev', 'i4'), ('length', 'f4')])
FACE_DTYPE = np.dtype([('halfedge', 'i4'), ('normal', '3f4'), ('area', 'f4')])
VERTEX_DTYPE = np.dtype([('position', '3f4'), ('normal', '3f4'), ('halfedge', 'i4'), ('valence', 'i4'), ('neighbors', '8i4')])

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
        self._loop_subdivision_flip_edges = []
        self._loop_subdivision_new_vertices = []

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
        Return the normal of each triangular face.
        """
        if np.all(self._faces['normal'] == -1):
            triangle_mesh_utils.c_update_face_normals(list(np.arange(len(self._faces)).astype(np.int32)), self._halfedges, self._vertices, self._faces)
            # v2 = self.vertices[self._halfedges['vertex'][self._halfedges['prev'][self._faces['halfedge']]]]
            # v1 = self.vertices[self._halfedges['vertex'][self._faces['halfedge']]]
            # v0 = self.vertices[self._halfedges['vertex'][self._halfedges['next'][self._faces['halfedge']]]]
            # u = v2 - v1
            # v = v0 - v1
            # n = np.cross(u, v, axis=1)
            # nn = np.linalg.norm(n, axis=1)
            # self._faces['area'] = 0.5*nn
            # self._faces['normal'] = n/nn[:, None]
            # self._faces['normal'][np.isnan(self._faces['normal'])] = 0
            # self._faces['normal'][self._faces['halfedge'] == -1] = -1
        return self._faces['normal']

    @property
    def face_areas(self):
        """
        Return the area of each triangular face.
        """
        if np.all(self._faces['area'] == -1):
            self._faces['normal'][:] = -1
            self.face_normals
        return self._faces['area']
    
    @property
    def vertex_neighbors(self):
        """
        Return the up to 6 neighbors of each vertex.
        """
        if np.all(self._vertices['neighbors'] == -1):
            triangle_mesh_utils.c_update_vertex_neighbors(list(np.arange(len(self._vertices)).astype(np.int32)), self._halfedges, self._vertices, self._faces)
            # for v_idx in np.arange(len(self.vertices)):
            #     _orig = self._vertices['halfedge'][v_idx]
            #     _curr = _orig

            #     curr_edge = self._halfedges[_curr]
            #     _twin = curr_edge['twin']
            #     twin_edge = self._halfedges[_twin]

            #     self._vertices['normal'][v_idx] = 0

            #     i = 0
            #     self._vertices['valence'][v_idx] = 0
            #     while True:
            #         if (_curr == -1) or (_twin == -1):
            #             break
            #         _vertex = curr_edge['vertex']
            #         _face = curr_edge['face']
            #         if (i < 8):
            #             self._vertices['neighbors'][v_idx, i] = _curr
            #             n = self._faces['normal'][_face]
            #             a = self._faces['area'][_face]
            #             self._vertices['normal'][v_idx] += n*a
                    
            #         l = self._vertices['position'][_vertex] - self._vertices['position'][self._halfedges['vertex'][curr_edge['prev']]]
            #         curr_edge['length'] = np.sqrt((l*l).sum())

            #         _curr = twin_edge['next']
            #         curr_edge = self._halfedges[_curr]
            #         _twin = curr_edge['twin']
            #         twin_edge = self._halfedges[_twin]

            #         i += 1
            #         if (_curr == _orig):
            #             break
            #     self._vertices['valence'][v_idx] = i
            # nn = np.linalg.norm(self._vertices['normal'], axis=1)
            # self._vertices['normal'] /= nn[:, None]
            # self._vertices['normal'][np.isnan(self._vertices['normal'])] = 0
            # self._vertices['normal'][self._vertices['halfedge'] == -1] = -1

        return self._vertices['neighbors']

    @property
    def vertex_normals(self):
        """
        Return the normal of each vertex.
        """
        if np.all(self._vertices['normal'] == -1):
            self._vertices['neighbors'][:] = -1
            self.vertex_neighbors
        return self._vertices['normal']

    @property
    def valences(self):
        """
        Return the valence of each vertex.
        """
        if np.all(self._vertices['valence'] == -1):
            self._vertices['neighbors'][:] = -1
            self.vertex_neighbors
        return self._vertices['valence']

    @property
    def edge_dict(self):
        """
        Return a dictionary of the edges in the mesh, each stored as a single number.
        """
        # edges = np.vstack([self.faces[:,[0,1]], self.faces[:,[1,2]], self.faces[:,[2,0]]])
        edges = np.vstack([self._halfedges['vertex'], self._halfedges[self._halfedges['twin']]['vertex']]).T
        
        d = {}
        for i, e in enumerate(pack_edges(edges)):
            d[e] = i
            
        return d

    @property
    def edge_valences(self):
        # edges = np.vstack([self.faces[:,[0,1]], self.faces[:,[1,2]], self.faces[:,[2,0]]])
        edges = np.vstack([self._halfedges['vertex'], self._halfedges[self._halfedges['twin']]['vertex']]).T

        packed_edges = pack_edges(edges)
        e, c = np.unique(packed_edges, return_counts=True)

        d = {}
        for k, v in zip(e, c):
            d[k] = v
        
        return d

    @property
    def manifold(self):
        """
        Checks if the mesh is manifold: Is every edge is shared by exactly two triangles?
        """
        edges = np.vstack([self.faces[:,[0,1]], self.faces[:,[1,2]], self.faces[:,[2,0]]])

        packed_edges = pack_edges(edges)
        _, c = np.unique(packed_edges, return_counts=True)
        
        return np.all(c==2)

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

        triangle_mesh_utils.c_update_face_normals(f_idxs, self._halfedges, self._vertices, self._faces)

        # for f_idx in f_idxs:
        #     v2 = self.vertices[self._halfedges['vertex'][self._halfedges['prev'][self._faces['halfedge'][f_idx]]]]
        #     v1 = self.vertices[self._halfedges['vertex'][self._faces['halfedge'][f_idx]]]
        #     v0 = self.vertices[self._halfedges['vertex'][self._halfedges['next'][self._faces['halfedge'][f_idx]]]]
        #     u = v2 - v1
        #     v = v0 - v1
        #     n = fast_3x3_cross(u.squeeze(), v.squeeze())
        #     nn = np.sqrt((n*n).sum())
        #     self._faces['area'][f_idx] = 0.5*nn
        #     if (self._faces['halfedge'][f_idx] == -1):
        #         self._faces['normal'][f_idx] = -1
        #     if (nn > 0):
        #         self._faces['normal'][f_idx] = n/nn
        #     else:
        #         self._faces['normal'][f_idx] = 0

    def _update_vertex_neighbors(self, v_idxs):
        """
        Recalculate len(v_idxs) vertex neighbors/normals.

        Parameters
        ----------
            v_idxs : list or np.array
                List of vertex indicies indicating which vertices to update.
        """

        triangle_mesh_utils.c_update_vertex_neighbors(v_idxs, self._halfedges, self._vertices, self._faces)
        
        # for v_idx in v_idxs:
        #     _orig = self._vertices['halfedge'][v_idx]
        #     _curr = _orig
            
        #     curr_edge = self._halfedges[_curr]
        #     _twin = curr_edge['twin']
        #     twin_edge = self._halfedges[_twin]
            
        #     if self.debug and (twin_edge['vertex'] != v_idx):
        #         print(_curr, curr_edge, _twin,twin_edge)
        #         raise RuntimeError('Twin (%d) should point back to starting vertex (%d) but points to %d' % (_twin, v_idx, twin_edge['vertex']))
            
        #     i = 0
        #     self._vertices['valence'][v_idx] = 0
        #     self._vertices['neighbors'][v_idx] = -1
        #     self._vertices['normal'][v_idx] = 0
            
        #     _normal = 0*self._vertices['normal'][v_idx]  # avoid a few lookups by using a local variable

        #     while True:
        #         if (_curr == -1) or (_twin == -1):
        #             break

        #         if (i < 8):
        #             self._vertices['neighbors'][v_idx, i] = _curr
                    
        #             #TODO - should these be in if clause?
        #             _face = curr_edge['face']
        #             n = self._faces['normal'][_face]
        #             a = self._faces['area'][_face]
        #             _normal += n*a
        #         else:
        #             pass
        #             if self.debug and (i > 20):
        #                 print('Abnormal vertex valance detected on vertex %d' % v_idx)
        #                 # raise RuntimeError('Abnormal vertex valance detected on vertex %d' % v_idx)
                
        #         vertex = self._vertices['position'][v_idx]
        #         l = vertex - self._vertices['position'][curr_edge['vertex']]
        #         l = np.sqrt((l*l).sum())
        #         curr_edge['length'] = l
        #         twin_edge['length'] = l
                
        #         _curr = twin_edge['next']
        #         curr_edge = self._halfedges[_curr]
        #         _twin = curr_edge['twin']
        #         twin_edge = self._halfedges[_twin]
                
        #         i += 1
                
        #         if (_curr == _orig):
        #             break

        #     self._vertices['valence'][v_idx] = i

        #     if self.debug and (self._valences[v_idx] < 3):
        #         raise RuntimeError('Detected valence <3 on vertex %d' % v_idx)

        #     nn = np.sqrt((_normal*_normal).sum())
        #     if nn > 0:
        #         self._vertices['normal'][v_idx] = _normal/nn
        #     else:
        #         self._vertices['normal'][v_idx] = 0

    def _resize(self, vec, axis=0, skip_entries=True, return_orig=False, key=None):
        """
        Increase the size of an input vector 1.5x, keeping all active data.

        Parameters
        ---------
            vec : np.array
                Vector to expand, -1 stored in unused entries.
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

        # Grab the valences of the 4 points near the edge
        vl, vd = self._vertices['valence'][_live_vertex], self._vertices['valence'][_dead_vertex]
        vn, vtn = self._vertices['valence'][self._halfedges['vertex'][_next]], self._vertices['valence'][self._halfedges['vertex'][_twin_next]]

        # Make sure we create no vertices of valence <3 (manifoldness)
        if ((vl + vd - 3) < 4) or (vn < 4) or (vtn < 4):
            return

        # Check for creation of multivalent edges and prevent this (manifoldness)
        dead_list = self._halfedges['vertex'][self._halfedges['twin'][self._halfedges['vertex'] == _dead_vertex]]
        live_list = self._halfedges['vertex'][self._halfedges['twin'][self._halfedges['vertex'] == _live_vertex]]
        twin_list = list(set(dead_list) & set(live_list))
        if len(twin_list) != 2:
            return
        
        if self.debug and (_live_vertex == _dead_vertex):
            print(_curr, curr_halfedge, _twin, twin_halfedge)
            raise RuntimeError('Live vertex equals dead vertex (both halves of edge point to same place)')

        # Collapse to the midpoint of the original edge vertices
        self._halfedges['vertex'][self._halfedges['vertex'] == _dead_vertex] = _live_vertex
        self._vertices['position'][_live_vertex, :] = 0.5*(self.vertices[_live_vertex, :] + self.vertices[_dead_vertex, :])
        
        # update valence of vertex we keep
        self._vertices['valence'][_live_vertex] = vl + vd - 3
        
        if self.debug:
            print(self._vertices['valence'][_live_vertex], self._vertices['valence'][_dead_vertex])
            assert(self._vertices['valence'][_live_vertex] >=3)
        
        # delete dead vertex
        self._vertices[_dead_vertex] = -1

        # Zipper the remaining triangles
        def _zipper(edge1, edge2):
            t1 = self._halfedges['twin'][edge1]
            t2 = self._halfedges['twin'][edge2]
            
            if (t2 != -1):
                self._halfedges['twin'][t2] = t1
                
            if (t1 != -1):
                self._halfedges['twin'][t1] = t2
                
        _zipper(_next, _prev)
        _zipper(_twin_next, _twin_prev)

        # We need some more pointers
        _prev_twin = self._halfedges['twin'][_prev]
        _prev_twin_vertex = self._halfedges['vertex'][_prev_twin]
        _next_prev_twin = self._halfedges['next'][_prev_twin]
        _next_prev_twin_vertex = self._halfedges['vertex'][_next_prev_twin]
        _twin_next_vertex = self._halfedges['vertex'][_twin_next]
        _next_twin_twin_next = self._halfedges['next'][self._halfedges['twin'][_twin_next]]
        _next_twin_twin_next_vertex = self._halfedges['vertex'][_next_twin_twin_next]

        # Make sure we have good _vertex_halfedges references
        self._vertices['halfedge'][_live_vertex] = _prev_twin
        self._vertices['halfedge'][_prev_twin_vertex] = _next_prev_twin
        self._vertices['halfedge'][_twin_next_vertex] = self._halfedges['twin'][_twin_next]
        self._vertices['halfedge'][_next_twin_twin_next_vertex] = self._halfedges['next'][_next_twin_twin_next]

        # Grab faces to update
        face0 = self._halfedges['face'][self._halfedges['twin'][_next]]
        face1 = self._halfedges['face'][self._halfedges['twin'][_prev]]
        face2 = self._halfedges['face'][self._halfedges['twin'][_twin_next]]
        face3 = self._halfedges['face'][self._halfedges['twin'][_twin_prev]]

        # Delete the inner triangles
        self._face_delete(_curr)
        self._face_delete(_twin)
        # _curr_face = curr_halfedge['face']
        # _twin_face = twin_halfedge['face']
        # self._faces[_curr_face] = -1
        # self._faces[_twin_face] = -1

        if self.debug:
            print(curr_halfedge, twin_halfedge)

        # # Delete curr, next, prev
        # self._edge_delete(_curr)
        # self._edge_delete(_prev)
        # self._edge_delete(_next)

        # # Delete _twin, _twin_prev, _twin_next
        # self._edge_delete(_twin)
        # self._edge_delete(_twin_prev)
        # self._edge_delete(_twin_next)

        try:
            if live_update:
                # Update faces
                self._update_face_normals([face0, face1, face2, face3])
                self._update_vertex_neighbors([_live_vertex, _prev_twin_vertex, _next_prev_twin_vertex, _twin_next_vertex])
    
                self._faces_by_vertex = None
        
        except RuntimeError as e:
            print(_curr, _twin, _next, _prev, _twin_next, _twin_prev, _next_prev_twin, _next_twin_twin_next, _prev_twin)
            print([_live_vertex, _prev_twin_vertex, _next_prev_twin_vertex, _twin_next_vertex])
            raise e

    def _face_delete(self, _edge):
        """
        Delete a face defined by the halfedge _edge.

        Parameters
        ----------
            edge : int
                One self._halfedge index of the edge defining a face to delete.
        """
        curr_edge = self._halfedges[_edge]
        self._faces[curr_edge['face']] = -1

        self._edge_delete(curr_edge['next'])
        self._edge_delete(curr_edge['prev'])
        self._edge_delete(_edge)

    def _edge_delete(self, _edge):
        """
        Delete edge _edge in place from self._halfedges.

        Parameters
        ----------
            edge : int
                One self._halfedge index of the edge to delete.
        """
        if _edge == -1:
            return
        
        self._halfedges[_edge] = -1
        self._halfedge_vacancies.append(_edge)

    def _insert(self, el, el_arr, el_vacancies, key, compact=False, **kwargs):
        """
        Insert an element into an array at the position of the smallest empty entry
        when searching the array by key.

        Parameters
        ----------
            el 
                Element to add to the array. Must be one of the dtypes in the structured 
                dtype of el_arr.
            el_arr : np.array
                Array where we add the element
            el_vacancies : list
                External list where we keep empty positions of el_arr.
            key : string
                Key on which to search the array
            compact : bool
                Do we copy -1 values in the resize of el_arr?
            kwargs 
                List of dtype parameters to additionally define el.

        Returns
        -------
            ed
                Element added, includes el stored in full structured dtype of el_arr.
            idx : int
                Index of el_arr where ed is stored.
            el_arr : np.array
                Modified array containing ed.
            el_vacancies : list
                Modified list of el_arr vacancies.
        """
        try:
            idx = el_vacancies.pop(0)
        except IndexError:
            # no vacant slot, resize
            el_arr = self._resize(el_arr, skip_entries=compact, key=key)
            #TODO - invalidate neighbours, vertex_halfedges etc ???
            
            # NOTE: If we search by a different key next time, el_vacancies will
            # still contain the vacant positions when searching on the previous key.
            if len(el_arr[key].shape) > 1:
                el_vacancies = [int(x) for x in np.argwhere(np.sum(el_arr[key],axis=1)/el_arr[key].shape[1] == -1)]
            else:
                el_vacancies = [int(x) for x in np.argwhere(el_arr[key] == -1)]

            idx = el_vacancies.pop(0)
        
        # Put el in el_arr
        ed = el_arr[idx]
        ed[key] = el
        # Define additional structured dtype values for el
        for k, v in kwargs.items():
            ed[k] = v
            
        return ed, idx, el_arr, el_vacancies

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

        Parameters
        ----------
            _edge : idx
                self._halfedges index of a single halfedge on this face
            compact : bool
                Do we copy -1 values in the resize of _faces?
            kwargs 
                List of _faces dtype parameters (e.g. area=13.556) to define face
                via more than just one of its halfedges.

        Returns
        -------
            fa : FACE_DTYPE
                Face created
            idx : int
                Index of the face in self._faces.
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
            compact : bool
                Do we copy -1 values in the resize of _vertices?
            kwargs 
                List of _vertices dtype parameters (e.g. valence=2) to define vertex
                via more than just its position.

        Returns
        -------
            vx : VERTEX_DTYPE
                Vertex created
            idx : int
                Index of the vertex in self._vertices.
        """
        vx, idx, self._vertices, self._vertex_vacancies = self._insert(_vertex, self._vertices, self._vertex_vacancies, 'position', compact, **kwargs)

        self._faces_by_vertex = None  # Reset

        return vx, idx

    def edge_split(self, _curr, live_update=True, loop_subdivide=False):
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
            loop_subdivide: bool
                Are we doing loop subdivision? If so, keep track of all edges
                incident on both a new vertex and an old verex that do not
                split an existing edge.
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
        self._faces['halfedge'][curr_edge['face']] = _curr
        self._faces['halfedge'][twin_edge['face']] = _twin
        _, _face_1_idx = self._new_face(_twin_prev)
        _, _face_2_idx = self._new_face(_next)
        self._halfedges['face'][_twin_prev] = _face_1_idx
        self._halfedges['face'][_next] = _face_2_idx

        # Insert the new faces
        _, _he_0_idx = self._new_edge(self._halfedges['vertex'][_next], prev=_curr, next=_prev, face=curr_edge['face'])
        _, _he_1_idx = self._new_edge(_vertex_idx, prev=_twin_next, next=_twin, face=twin_edge['face'])
        
        _, _he_2_idx = self._new_edge(self._halfedges['vertex'][_twin_next], next=_twin_prev, face=_face_1_idx)
        _, _he_3_idx = self._new_edge(_vertex_idx, prev=_twin_prev, next=_he_2_idx, face=_face_1_idx)
        self._halfedges['prev'][_he_2_idx] = _he_3_idx

        _, _he_4_idx = self._new_edge(curr_edge['vertex'], next=_next, face=_face_2_idx)
        _, _he_5_idx = self._new_edge(_vertex_idx, prev=_next, next=_he_4_idx, face=_face_2_idx)
        self._halfedges['prev'][_he_4_idx] = _he_5_idx

        self._halfedges['twin'][_he_0_idx] = _he_5_idx
        self._halfedges['twin'][_he_5_idx] = _he_0_idx

        self._halfedges['twin'][_he_1_idx] = _he_2_idx
        self._halfedges['twin'][_he_2_idx] = _he_1_idx

        self._halfedges['twin'][_he_3_idx] = _he_4_idx
        self._halfedges['twin'][_he_4_idx] = _he_3_idx

        # Make sure old vertices have existing vertex halfedges
        self._vertices['halfedge'][curr_edge['vertex']] = _he_3_idx
        self._vertices['halfedge'][twin_edge['vertex']] = _curr
        self._vertices['halfedge'][self._halfedges['vertex'][_next]] = _he_5_idx
        self._vertices['halfedge'][self._halfedges['vertex'][_twin_next]] = _he_1_idx

        # Update _prev, next
        self._halfedges['prev'][_prev] = _he_0_idx
        self._halfedges['prev'][_next] = _he_4_idx
        self._halfedges['next'][_next] = _he_5_idx

        # Update _twin_next, _twin_prev
        self._halfedges['next'][_twin_next] = _he_1_idx
        self._halfedges['prev'][_twin_prev] = _he_2_idx
        self._halfedges['next'][_twin_prev] = _he_3_idx

        # Update _curr and _twin vertex
        self._halfedges['vertex'][_curr] = _vertex_idx
        self._vertices['halfedge'][_vertex_idx] = _twin
        self._halfedges['next'][_curr] = _he_0_idx
        self._halfedges['prev'][_twin] = _he_1_idx

        if loop_subdivide:
            # Make sure these edges emanate from the new vertex stored at _vertex_idx
            self._loop_subdivision_flip_edges.extend([_he_0_idx, _he_2_idx])
            self._loop_subdivision_new_vertices.extend([_vertex_idx])

        if live_update:
            self._update_face_normals([self._halfedges['face'][_he_0_idx], self._halfedges['face'][_he_1_idx], self._halfedges['face'][_he_2_idx], self._halfedges['face'][_he_4_idx]])
            self._update_vertex_neighbors([curr_edge['vertex'], twin_edge['vertex'], self._halfedges['vertex'][_he_0_idx], self._halfedges['vertex'][_he_2_idx], self._halfedges['vertex'][_he_4_idx]])

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

        curr_edge = self._halfedges[_curr]
        _prev = curr_edge['prev']
        _next = curr_edge['next']
        _twin = curr_edge['twin']
        
        twin_edge = self._halfedges[_twin]
        _twin_prev = twin_edge['prev']
        _twin_next = twin_edge['next']

        # Make sure both vertices have valence > 3 (preserve manifoldness)
        if (self._valences[curr_edge['vertex']] < 4) or (self._valences[twin_edge['vertex']] < 4):
            return

        # Calculate adjustments to the halfedges we're flipping
        new_v0 = self._halfedges['vertex'][_next]
        new_v1 = self._halfedges['vertex'][_twin_next]

        # If there's already an edge between these two vertices, don't flip (preserve manifoldness)
        if new_v1 in self._halfedges['vertex'][self._halfedges['twin'][self._halfedges['vertex'] == new_v0]]:
            return

        # Convexity check
        y = self._vertices['position'][self._halfedges['vertex'][_curr]] - self._vertices['position'][self._halfedges['vertex'][_twin]]
        x = self._vertices['position'][new_v1] - self._vertices['position'][new_v0]
        xn = x/np.sqrt((x*x).sum())
        yn = y/np.sqrt((y*y).sum())
        p = np.eye(3) - xn[:,None]*yn[None,:]
        z = self._vertices['position'][self._halfedges['vertex'][_twin]] - self._vertices['position'][new_v0]
        pz = (p*z).sum(1)
        y_x_pz = fast_3x3_cross(x,pz)
        n = fast_3x3_cross(x,y)
        if (y_x_pz*n).sum() > 0:
            # Concave, don't flip
            return

        # # Make sure the dihedral angle between the flipped triangles is less than (2/3)*pi
        # upper = self._vertices['position'][self._halfedges['vertex'][_curr]] - self._vertices['position'][new_v0]
        # lower = self._vertices['position'][self._halfedges['vertex'][_twin]] - self._vertices['position'][new_v0]
        # new_edge = self._vertices['position'][new_v1] - self._vertices['position'][new_v0]
        # n1 = fast_3x3_cross(new_edge, upper)
        # n2 = fast_3x3_cross(lower, new_edge)
        # nn1 = n1/np.sqrt((n1*n1).sum())
        # nn2 = n2/np.sqrt((n2*n2).sum())
        # theta = np.arccos(np.sqrt((nn1*nn2).sum()))
        # if theta >= 2.094:
        #     return

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
        # vertex_halfedges references to halfedges we know work
        self._vertices['halfedge'][curr_edge['vertex']] = _next
        self._vertices['halfedge'][twin_edge['vertex']] = _twin_next

        # Apply adjustments to the the halfedges we're flipping
        curr_edge['vertex'] = new_v0
        twin_edge['vertex'] = new_v1
        curr_edge['next'] = _prev
        twin_edge['next'] = _twin_prev
        curr_edge['prev'] = _twin_next
        twin_edge['prev'] = _next

        # Update pointers
        _prev = curr_edge['prev']
        _next = curr_edge['next']
        _twin_prev = twin_edge['prev']
        _twin_next = twin_edge['next']

        # Set faces
        self._halfedges['face'][_next] = curr_edge['face']
        self._halfedges['face'][_prev] = curr_edge['face']
        self._halfedges['face'][_twin_next] = twin_edge['face']
        self._halfedges['face'][_twin_prev] = twin_edge['face']

        # Finish updating vertex_halfedges references, update face references
        self._vertices['halfedge'][curr_edge['vertex']] = _twin
        self._vertices['halfedge'][twin_edge['vertex']] = _curr
        self._faces['halfedge'][curr_edge['face']] = _curr
        self._faces['halfedge'][twin_edge['face']] = _twin

        if live_update:
            # Update face and vertex normals
            self._update_face_normals([curr_edge['face'], twin_edge['face']])
            self._update_vertex_neighbors([curr_edge['vertex'], twin_edge['vertex'], self._halfedges['vertex'][_next], self._halfedges['vertex'][_twin_next]])

            self._faces_by_vertex = None

    def regularize(self):
        """
        Adjust vertices so they tend toward valence 6.
        """
    
        j = 0
        while (j < 17) and max(self._valences > 6):
            j += 1
        
            # Find which vertices have high valences
            problems = np.where(self._valences > 6)[0]
            delta0 = np.abs(self._valences[problems] - 6)
                        
            # Find max-valence incident vertex
            neighbours = self._vertex_neighbors[problems]
            neighbour_vertices = self._halfedges[neighbours]['vertex']
            _target_valence_n_id = np.argmax(self._valences[neighbour_vertices], axis=1)
            _target_valence_n_e = neighbours[np.arange(len(_target_valence_n_id)), _target_valence_n_id]
            
            e_to_flip = _target_valence_n_e
        
            # Find low-valence vertices
            p1 = np.where((self._valences > 0) & (self._valences < 6))[0]
            delta1 = np.abs(self._valences[p1] - 6)
        
            deltas = np.hstack((delta0, delta1))
            edges_to_flip = np.hstack((e_to_flip, self._halfedges[self._vertex_halfedges[p1]]['next']))

            # Flip the worst offenders first
            edges_to_flip = edges_to_flip[np.argsort(deltas)[::-1]]
            
            ef = list(edges_to_flip)
            k = 0
            while (k < len(ef)):
                e = ef[k]
                try:
                    ef.remove(self._halfedges[e]['twin'])
                except ValueError:
                    pass
                k += 1
                
            edges_to_flip = np.array(ef)
        
            # print(len(edges_to_flip), len(problems), len(p1))
            
            # Loop over high valence vertices and flip edges to reduce valence
            for _idx in edges_to_flip:
                i = 0
                if (self._valences[self._halfedges[_idx]['vertex']] > (6)):
                    self.edge_flip(_idx, live_update=True)
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
            # an = 0.5*np.linalg.norm(np.cross(np.diff(fc, axis=1), (self._vertices[:,None,:]-fc)[:,:(6-1),:], axis=2), axis=2)*nn_mask
            an = (1./self._halfedges['length'][nn])*nn_mask
            an[self._halfedges['length'][nn] == 0] = 0

            # Calculate gravity-weighted centroids
            A = np.sum(an, axis=1)
            c = 1./(A[...,None])*np.sum(an[...,None]*vn, axis=1)
            # Don't get messed up by slivers. NOTE: This is a hack in case
            # edge split/collapse don't do their jobs.
            A_mask = (A == 0)
            c[A_mask] = self._vertices['position'][A_mask]

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

    def remesh(self, n=5, target_edge_length=-1, l=0.5, n_relax=10):
        """
        Produce a higher-quality mesh.

        Follows procedure in Botsch and Kobbelt, A Remeshing Approach 
        to Multiresoluton Modeling, Eurographics Symposium on Geometry 
        Processing, 2004.

        Parameters
        ----------
            n : int
                Number of remeshing iterations to apply.
            target_edge_length : float
                Target edge length for all edges in the mesh.
            l : float
                Relaxation regularization term, used to avoid oscillations.
            n_relax : int 
                Number of Lloyd relaxation (relax()) iterations to apply 
                per remeshing iteration.
        """

        mean_edge_length = np.mean(self._halfedges['length'][self._halfedges['length'] != -1])

        if (target_edge_length == -1):
            # Guess edge_length
            target_edge_length = mean_edge_length

        if (target_edge_length <= 0.25*mean_edge_length):
            # Apply loop subdivision first.
            # print('Upsampling...')
            n_iters = int(mean_edge_length/target_edge_length/4. + 0.5)
            self.loop_subdivide(n_iters)

        for k in range(n):
            # 1. Split all edges longer than (4/3)*target_edge_length at their midpoint.
            split_count = 0
            for i in np.arange(len(self._halfedges['length'])):
                if (self._halfedges['length'][i] >= 0) and (self._halfedges['length'][i] > 1.33*target_edge_length):
                    self.edge_split(i)
                    split_count += 1
            print('Split count: %d' % (split_count))
            
            if self.debug:
                # check for dud half edges
                _valid = self._halfedges['vertex'] != -1
                assert(not np.any(self._halfedges['twin'][_valid] == -1))
                assert (not np.any(self._halfedges['next'][_valid] == -1))
                assert (not np.any(self._halfedges['vertex'][self._halfedges['next'][_valid]] == -1))

                assert (not np.any(self._halfedges['next'][self._halfedges['twin'][_valid]] == self._halfedges['twin'][self._halfedges['prev'][_valid]]))
                
                assert(self._vertices['valence'][self._vertices['valence']>0].min() >=3)
        
            # 2. Collapse all edges shorter than (4/5)*target_edge_length to their midpoint.
            collapse_count = 0
            for i in np.arange(len(self._halfedges['length'])):
                if (self._halfedges['length'][i] >=0) and (self._halfedges['length'][i] < 0.8*target_edge_length):
                    self.edge_collapse(i)
                    collapse_count += 1
            print('Collapse count: ' + str(collapse_count))

            # 3. Flip edges in order to minimize deviation from 6.
            self.regularize()

            # 4. Relocate vertices on the surface by tangential smoothing.
            # self.relax(l=l, n=n_relax)

    def repair(self):
        """
        Make the mesh manifold.
        """

        # # Delete all edges with valence not equal to 2
        # edges = np.vstack([self._halfedges['vertex'], self._halfedges[self._halfedges['twin']]['vertex']]).T

        # packed_edges = pack_edges(edges)
        # e, c = np.unique(packed_edges, return_counts=True)

        # d = {}
        # for k, v in zip(e, c):
        #     if v == 2:
        #         continue
        #     d[k] = v

        # for i, e in enumerate(packed_edges):
        #     if e in list(d.keys()):
        #         self._face_delete(i)

        # # Find halfedges forming negative triangle
        # no_twin = (self._halfedges['twin'] == -1)
        # _edge_0, _edge_1 = np.where((self._halfedges['vertex'][:,None] == self._halfedges['vertex'][self._halfedges['prev']][None, :])*no_twin[:,None]*no_twin[None,:])

        # for _arriving, _leaving in zip(_edge_0, _edge_1):
        #     arriving_edge = self._halfedges[_arriving]
        #     leaving_edge = self._halfedges[_leaving]

        #     v1 = self._halfedges['vertex'][arriving_edge['prev']]

        #     _, _he_0_idx = self._new_edge(arriving_edge['vertex'], twin=_leaving)
        #     _, _face_0_idx = self._new_face(_he_0_idx)
        #     _, _he_1_idx = self._new_edge(v1, twin=_arriving, prev=_he_0_idx, face=_face_0_idx)
        #     _, _he_2_idx = self._new_edge(leaving_edge['vertex'], prev=_he_1_idx, next=_he_0_idx, face=_face_0_idx)
            
        #     self._halfedges['face'][_he_0_idx] = _face_0_idx
        #     self._halfedges['next'][_he_0_idx] = _he_1_idx
        #     self._halfedges['prev'][_he_0_idx] = _he_2_idx
        #     self._halfedges['next'][_he_1_idx] = _he_2_idx

        #     self._update_face_normals([_face_0_idx])
        #     self._update_vertex_neighbors([arriving_edge['vertex'], leaving_edge['vertex'], v1])

        # pass

    def loop_subdivide(self, n=1):
        """
        Upsample the mesh by a factor of 4 (factor of 2 along each axis in the plane of the mesh).

        References:
            1. C. T. Loop, "Smooth Subdivision Surfaces Based on Triangles," University of Utah, 1987
            2. http://462cmu.github.io/asst2_meshedit/, task 4

        Parameters
        ----------
            n : int
                Number of upsampling operations to perform.
        """

        for k in range(n):
            # 1. Split every edge in the mesh
            split_edges = {}
            edges_to_split = np.where(self._halfedges['length'] != -1)[0]  # Mark the original edges
            # NOTE: We need this expensive operation so as to not split edges we create during the
            # other edge splits (e.g. if there are -1s interspersed in self._halfedges).
            for i in edges_to_split:
                if (i in list(split_edges.keys())):
                    continue
                
                split_edges[self._halfedges['twin'][i]] = i
                self.edge_split(i, loop_subdivide=True)
            
            # 2. Flip any new edge that touches an old vertex and a new vertex
            for i in np.arange(len(self._loop_subdivision_flip_edges)):
                e = self._loop_subdivision_flip_edges.pop()  # Empties self._loop_subdivision_flip_edges
                if (self._halfedges['vertex'][e] in self._loop_subdivision_new_vertices):
                    # We're trying to flip an edge connected by two new vertices.
                    continue
                self.edge_flip(e)

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