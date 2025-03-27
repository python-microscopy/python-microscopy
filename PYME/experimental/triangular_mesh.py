# -*- coding: utf8 -*-

import numpy as np

# Regular, boundary, and max vertex valences. These need tp be
VALENCE = 6
BOUNDARY_VALENCE = 4
NEIGHBORSIZE = 20  # Note this must match NEIGHBORSIZE in triangle_mesh_utils.h

USE_C = True

if USE_C:
    from PYME.experimental import triangle_mesh_utils

# def pack_edges(arr, axis=1):
#     """
#     Stores edges as a single unique integer. 
    
#     NOTE: that this implictly caps the mesh at 2**32
#     vertices.
#     """
#     arr = np.sort(arr, axis=axis)
#     res = ((arr[:,0].astype(np.uint64)) << 32)
#     res += arr[:,1].astype(np.uint64)
    
#     return res

MAX_VERTEX_COUNT = 2**64

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

HALFEDGE_DTYPE = np.dtype([('vertex', 'i4'), ('face', 'i4'), ('twin', 'i4'), ('next', 'i4'), ('prev', 'i4'), ('length', 'f4'), ('component', 'i4')])
FACE_DTYPE = np.dtype([('halfedge', 'i4'), ('normal', '3f4'), ('area', 'f4'), ('component', 'i4')])
# Note that VERTEX_DTYPE neighbors size must match NEIGHBORSIZE
VERTEX_DTYPE = np.dtype([('position', '3f4'), ('normal', '3f4'), ('halfedge', 'i4'), ('valence', 'i4'), ('neighbors', '20i4'), ('component', 'i4')])

LOOP_ALPHA_FACTOR = (np.log(13)-np.log(3))/12

class TriangleMesh(object):
    def __init__(self, vertices=None, faces=None, mesh=None, **kwargs):
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
        if mesh is not None:
            import copy
            
            self._vertices = np.copy(mesh._vertices)
            self._vertex_vacancies = copy.copy(mesh._vertex_vacancies)

            self._faces = np.copy(mesh._faces)
            self._face_vacancies = copy.copy(mesh._face_vacancies)

            self._halfedges = np.copy(mesh._halfedges)
            self._halfedge_vacancies = copy.copy(mesh._halfedge_vacancies)
        else:
            self._vertices = np.zeros(vertices.shape[0], dtype=VERTEX_DTYPE)
            self._vertices[:] = -1  # initialize everything to -1 to start with
            self._vertices['position'] = vertices
            self._vertex_vacancies = []

            self._faces = None  # Contains a pointer to one halfedge associated with each face
            self._face_vacancies = []

            # Halfedges
            self._halfedges = None
            self._halfedge_vacancies = []
            
            print('initializing halfedges ...')
            print('vertices.shape = %s, faces.shape = %s' % (vertices.shape, faces.shape))
            if vertices.shape[0] >= MAX_VERTEX_COUNT:
                raise RuntimeError('Maximum vertex count is %d, mesh has %d' % (MAX_VERTEX_COUNT, vertices.shape[0]))
            self._initialize_halfedges(vertices, faces)
            print('done initializing halfedges')

        # Representation of faces by triplets of vertices
        self._faces_by_vertex = None

        # loop subdivision vars
        self._loop_subdivision_flip_edges = []
        self._loop_subdivision_new_vertices = []

        # Singular edges
        self._singular_edges = None
        self._singular_vertices = None

        # Populate the normals
        self.face_normals
        self.vertex_normals

        # Properties we can visualize
        self.vertex_properties = ['x', 'y', 'z', 'component', 'boundary', 'singular']

        self.fix_boundary = True  # Hold boundary edges in place
        self.debug = False  # Print debug statements

        # Is the mesh manifold?
        self._manifold = None

        # Set debug, fix_boundary, etc.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, k):
        # this defers evaluation of the properties until we actually access them, 
        # as opposed to the mappings which stored the values on class creation.
        try:
            res = getattr(self, k)
        except AttributeError:
            raise KeyError('Key %s not defined' % k)
        
        return res

    def __copy__(self):
        return type(self)(mesh=self)

    @classmethod
    def from_stl(cls, filename, **kwargs):
        """
        Read from an STL file.
        """
        from PYME.IO.FileUtils import stl

        # Load an STL from file
        triangles_stl = stl.load_stl_binary(filename)

        # Call from_np_stl on the file stream
        return cls.from_np_stl(triangles_stl, **kwargs)

    @classmethod
    def from_ply(cls, filename, **kwargs):
        """
        Read from PLY file.
        """
        from PYME.IO.FileUtils import ply

        # Load a PLY from file
        vertices, faces, _ = ply.load_ply(filename)

        return cls(vertices, faces, **kwargs)

    @classmethod
    def from_np_stl(cls, triangles_stl, **kwargs):
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

        print('Data munged to vertices, faces')
        return cls(vertices, faces, **kwargs)

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
    def component(self):
        # if np.all(self._vertices['component'] == -1):
        self.find_connected_components()
        return self._vertices['component']

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
    def boundary(self):
        nn = self._vertices['neighbors']
        nn_mask = (nn!=-1)
        return np.any((self._halfedges['twin'][nn] == -1)*nn_mask, axis=1)

    @property
    def singular(self):
        self._singular_edges = None
        self._singular_vertices = None
        singular_vertices = np.zeros_like(self._vertices['halfedge'])
        if len(self.singular_vertices) > 0:
            singular_vertices[np.array(self.singular_vertices)] = 1
        return singular_vertices

    @property
    def singular_edges(self):
        """
        Get a list of singular edges. A singular edge is any edge shared by
        more than two faces.
        """
        if self._singular_edges is None:
            edges = np.vstack([self._halfedges['vertex'], self._halfedges[self._halfedges['prev']]['vertex']]).T
            edges = edges[edges[:,0] != -1]  # Drop non-edges
            # packed_edges = pack_edges(edges)
            packed_edges = np.sort(edges, axis=1)
            e, c = np.unique(packed_edges, return_counts=True, axis=0)
            singular_packed_edges = e[c>2]
            singular_edges = []
            for singular_packed_edge in singular_packed_edges:
                singular_edges.extend(list(np.where(packed_edges[:, None] == singular_packed_edge)[0]))

            singular_edges.extend(self._halfedges['twin'][singular_edges])
            self._singular_edges = list(set(singular_edges) - set([-1]))

        return self._singular_edges

    @property
    def singular_vertices(self):
        """
        Get a list of singular vertices. This includes endpoints of singular 
        edges, and isolated singular vertices. Isolated singular vertices
        have more than one 1-neighbor ring.
        """
        if self._singular_vertices is None:
            # Mark vertices that are endpoints of these edges
            singular_vertices = list(set(list(np.hstack([self._halfedges['vertex'][self.singular_edges], self._halfedges['vertex'][self._halfedges['twin'][self.singular_edges]]]))))
            
            # Mark isolated singular vertices
            for _vertex in np.arange(self._vertices.shape[0]):
                if self._vertices['halfedge'][_vertex] == -1:
                    continue

                # If the number of elements in the 1-neighbor ring does not match 
                # the number of halfedges pointing to a vertex, we have an isolated
                # singular vertex (note this requires the C code version of
                # update_vertex_neighbors, which reverses direction upon hitting
                # an edge to ensure a more accurate neighborhood)
                n_incident = np.flatnonzero(self._halfedges['vertex'] == _vertex).size
                n_neighbors = np.flatnonzero(self._vertices['neighbors'][_vertex] != -1).size
                if n_neighbors != n_incident:
                    singular_vertices.append(_vertex)

            # Remove duplicates
            self._singular_vertices = list(set(singular_vertices) - set([-1]))

        return self._singular_vertices

    @property
    def face_normals(self):
        """
        Return the normal of each triangular face.
        """
        if np.all(self._faces['normal'] == -1):
            if USE_C:
                triangle_mesh_utils.c_update_face_normals(list(np.arange(len(self._faces)).astype(np.int32)), self._halfedges, self._vertices, self._faces)
            else:
                v2 = self._vertices['position'][self._halfedges['vertex'][self._halfedges['prev'][self._faces['halfedge']]]]
                v1 = self._vertices['position'][self._halfedges['vertex'][self._faces['halfedge']]]
                v0 = self._vertices['position'][self._halfedges['vertex'][self._halfedges['next'][self._faces['halfedge']]]]
                u = v2 - v1
                v = v0 - v1
                n = np.cross(u, v, axis=1)
                nn = np.linalg.norm(n, axis=1)
                self._faces['area'] = 0.5*nn
                self._faces['normal'] = n/nn[:, None]
                self._faces['normal'][np.isnan(self._faces['normal'])] = 0
                self._faces['normal'][self._faces['halfedge'] == -1] = -1
        return self._faces['normal'][self._faces['halfedge'] != -1]

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
        Return the up to NEIGHBORSIZE neighbors of each vertex.
        """
        if np.all(self._vertices['neighbors'] == -1):
            if USE_C:
                triangle_mesh_utils.c_update_vertex_neighbors(list(np.arange(len(self._vertices)).astype(np.int32)), self._halfedges, self._vertices, self._faces)
            else:
                for v_idx in np.arange(len(self._vertices)):
                    _orig = self._vertices['halfedge'][v_idx]
                    _curr = _orig

                    curr_edge = self._halfedges[_curr]
                    _twin = curr_edge['twin']
                    twin_edge = self._halfedges[_twin]

                    self._vertices['normal'][v_idx] = 0

                    i = 0
                    self._vertices['valence'][v_idx] = 0
                    while True:
                        if _curr == -1:
                            break
                        _vertex = curr_edge['vertex']
                        _face = curr_edge['face']
                        if (i < NEIGHBORSIZE):
                            self._vertices['neighbors'][v_idx, i] = _curr
                            n = self._faces['normal'][_face]
                            a = self._faces['area'][_face]
                            self._vertices['normal'][v_idx] += n*a
                        
                        l = self._vertices['position'][_vertex] - self._vertices['position'][self._halfedges['vertex'][curr_edge['prev']]]
                        curr_edge['length'] = np.sqrt((l*l).sum())

                        _curr = twin_edge['next']
                        curr_edge = self._halfedges[_curr]
                        if _twin == -1:
                            break
                        _twin = curr_edge['twin']
                        twin_edge = self._halfedges[_twin]

                        i += 1
                        if (_curr == _orig):
                            break
                    self._vertices['valence'][v_idx] = i
                nn = np.linalg.norm(self._vertices['normal'], axis=1)
                self._vertices['normal'] /= nn[:, None]
                self._vertices['normal'][np.isnan(self._vertices['normal'])] = 0
                self._vertices['normal'][self._vertices['halfedge'] == -1] = -1

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
    def manifold(self):
        """
        Checks if the mesh is manifold: Is every edge is shared by exactly two 
        triangles?

        TODO: Check that we don't have any isolated singular vertices, which 
        also make the mesh non-manifold.
        """
        # edges = np.vstack([self._halfedges['vertex'], self._halfedges[self._halfedges['prev']]['vertex']]).T
        # edges[edges[:, 0] != -1]
        # edges = np.vstack([self.faces[:,[0,1]], self.faces[:,[1,2]], self.faces[:,[2,0]]])
        # packed_edges = pack_edges(edges)
        # _, c = np.unique(packed_edges, return_counts=True)
        
        # return np.all(c==2)

        if self._manifold is None:
            self._singular_edges = None
            self._singular_vertices = None
            vertex_mask = (self._halfedges['vertex'] != -1)
            twin_mask = (self._halfedges['twin'] == -1)
            boundary_check = ((np.flatnonzero(vertex_mask & twin_mask).size) == 0)
            self._manifold = (len(self.singular_vertices) == 0) & boundary_check

        return self._manifold

    def keys(self):
        return list(self.vertex_properties)

    def _initialize_halfedges(self, vertices, faces):
        """
        Accepts unordered vertices, indices parameterization of a triangular 
        mesh from an STL file and converts to topologically-connected halfedge
        representation.

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
            # edges_packed = pack_edges(edges)

            # Sort the edges lo->hi so we can arrange them uniquely
            # Convert to list of tuples for use with dictionary
            edges_packed = [tuple(e) for e in np.sort(edges, axis=1)]

            # Account for multivalent edges
            # edges_unique, edges_counts = np.unique(edges, return_counts=True, axis=0)

            # print('iterating edges')
            # # Use a dictionary to keep track of which edges are already assigned twins
            # d = {}
            # multivalent_edge = False
            # multivalent_dict = {}
            # for i, e in enumerate(edges_packed):
            #     if e in d:
            #         idx = d.pop(e)
            #         if self._halfedges['vertex'][idx] == self._halfedges['vertex'][i]:
            #             # Push it back in the queue and let's see if the next time we
            #             # pull it out we get the right matching edge.
            #             d[e] = idx
            #             # Trip the flag
            #             multivalent_edge = True
            #             multivalent_dict[e] = i
            #             continue
                        
            #         self._halfedges['twin'][idx] = i
            #         self._halfedges['twin'][i] = idx
            #     else:
            #         d[e] = i

            print('iterating edges')
            # Use a dictionary to keep track of which edges are already assigned twins
            d = {}
            for i, e in enumerate(edges_packed):
                if e in d:
                    idx = d.pop(e)
                    if self._halfedges['vertex'][idx] == self._halfedges['vertex'][i]:
                        # Don't assign a halfedge to multivalent edges
                        continue
                        
                    self._halfedges['twin'][idx] = i
                    self._halfedges['twin'][i] = idx
                else:
                    d[e] = i

            # print('multivalent')
            # # Go back and grab the remaining multivalent edges
            # if multivalent_edge:
            #     n_multivalent = len(list(multivalent_dict.keys()))
            #     print('%d multivalent edges' % n_multivalent)
            #     for j_, k in enumerate(list(multivalent_dict.keys())):
            #         if (not (j_ % 10)):
            #             print('mv edge %d of %d' % (j_, n_multivalent))
            #
            #         idx = multivalent_dict.pop(k)
            #         for i, e in enumerate(edges_packed):
            #             if e != idx:
            #                 continue
            #
            #             if self._halfedges['vertex'][idx] == self._halfedges['vertex'][i]:
            #                 continue
            #
            #             self._halfedges['twin'][idx] = i
            #             self._halfedges['twin'][i] = idx
            #             break
            #
            # print('multivalent done')
            
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

        if USE_C:
            if isinstance(f_idxs, list):
                f_idxs = np.int32(f_idxs)
            triangle_mesh_utils.c_update_face_normals(f_idxs, self._halfedges, self._vertices, self._faces)
        else:
            for f_idx in f_idxs:
                v2 = self._vertices['position'][self._halfedges['vertex'][self._halfedges['prev'][self._faces['halfedge'][f_idx]]]]
                v1 = self._vertices['position'][self._halfedges['vertex'][self._faces['halfedge'][f_idx]]]
                v0 = self._vertices['position'][self._halfedges['vertex'][self._halfedges['next'][self._faces['halfedge'][f_idx]]]]
                u = v2 - v1
                v = v0 - v1
                n = fast_3x3_cross(u.squeeze(), v.squeeze())
                nn = np.sqrt((n*n).sum())
                self._faces['area'][f_idx] = 0.5*nn
                if (self._faces['halfedge'][f_idx] == -1):
                    self._faces['normal'][f_idx] = -1
                if (nn > 0):
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

        if USE_C:
            if isinstance(v_idxs, list):
                v_idxs = np.int32(v_idxs)
            triangle_mesh_utils.c_update_vertex_neighbors(v_idxs, self._halfedges, self._vertices, self._faces)
        else:
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
                self._vertices['valence'][v_idx] = 0
                self._vertices['neighbors'][v_idx] = -1
                self._vertices['normal'][v_idx] = 0
                
                _normal = 0*self._vertices['normal'][v_idx]  # avoid a few lookups by using a local variable

                while True:
                    if (_curr == -1):
                        break

                    if (i < NEIGHBORSIZE):
                        self._vertices['neighbors'][v_idx, i] = _curr
                        
                        #TODO - should these be in if clause?
                        _face = curr_edge['face']
                        n = self._faces['normal'][_face]
                        a = self._faces['area'][_face]
                        _normal += n*a
                    else:
                        pass
                        if self.debug and (i > 2*NEIGHBORSIZE):
                            print('Abnormal vertex valance detected on vertex %d' % v_idx)
                            # raise RuntimeError('Abnormal vertex valance detected on vertex %d' % v_idx)
                    
                    vertex = self._vertices['position'][v_idx]
                    l = vertex - self._vertices['position'][curr_edge['vertex']]
                    l = np.sqrt((l*l).sum())
                    self._halfedges['length'][_curr] = l
                    if (_twin == -1):
                        break
                    self._halfedges['length'][_twin] = l
                    # curr_edge['length'] = l
                    # twin_edge['length'] = l
                    
                    _curr = twin_edge['next']
                    curr_edge = self._halfedges[_curr]
                    _twin = curr_edge['twin']
                    twin_edge = self._halfedges[_twin]
                    
                    i += 1
                    
                    if (_curr == _orig):
                        break

                self._vertices['valence'][v_idx] = i

                if self.debug and (self._vertices['valence'][v_idx] < 3):
                    raise RuntimeError('Detected valence <3 on vertex %d' % v_idx)

                nn = np.sqrt((_normal*_normal).sum())
                if nn > 0:
                    self._vertices['normal'][v_idx] = _normal/nn
                else:
                    self._vertices['normal'][v_idx] = 0

    def _resize(self, vec, axis=0, skip_entries=False, return_orig=False, key=None):
        """
        Increase the size of an input vector 1.5x, keeping all active data.

        Parameters
        ---------
            vec : np.array
                Vector to expand, -1 stored in unused entries.
            axis : int
                Axis (0 or 1) along which to resize vector (only works for 2D 
                arrays).
            skip_entries : bool
                Don't copy -1 entries to the new array.
            return_orig : bool
                Return indices of non-negative-one entries in the array prior 
                to resize.
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
                copy_size = np.flatnonzero(copy_mask).size
            else:
                copy_size = np.flatnonzero(copy_mask).size

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

        if _curr == -1:
            return

        # Create the pointers we need
        curr_halfedge = self._halfedges[_curr]
        _prev = curr_halfedge['prev']
        _next = curr_halfedge['next']
        _twin = curr_halfedge['twin']

        if (self._halfedges['twin'][_next] == -1) or (self._halfedges['twin'][_prev] == -1):
            # Collapsing this edge will create another free edge
            return

        interior = (_twin != -1)  # Are we on a boundary?

        if interior:
            nn = self._vertices['neighbors'][curr_halfedge['vertex']]
            nn_mask = (nn != -1)
            if (-1 in self._halfedges['twin'][nn]*nn_mask):
                return

            nn = self._vertices['neighbors'][self._halfedges['vertex'][_twin]]
            nn_mask = (nn != -1)
            if (-1 in self._halfedges['twin'][nn]*nn_mask):
                return

            twin_halfedge = self._halfedges[_twin]
            _twin_prev = twin_halfedge['prev']
            _twin_next = twin_halfedge['next']

            if self.debug:
                assert(_twin_prev != -1)
                assert(_twin_next != -1)

            if (self._halfedges['twin'][_twin_prev] == -1) or (self._halfedges['twin'][_twin_next] == -1):
                # Collapsing this edge will create another free edge
                return 

        _dead_vertex = self._halfedges['vertex'][_prev]
        _live_vertex = curr_halfedge['vertex']

        # Grab the valences of the 4 points near the edge
        if interior:
            vn, vtn = self._vertices['valence'][self._halfedges['vertex'][_next]], self._vertices['valence'][self._halfedges['vertex'][_twin_next]]

            # Make sure we create no vertices of valence <3 (manifoldness)
            # ((vl + vd - 3) < 4) or 
            if (vn < 4) or (vtn < 4):
                return

        vl, vd = self._vertices['valence'][_live_vertex], self._vertices['valence'][_dead_vertex]
        
        if ((vl + vd - 3) < 4):
            return

        # Check for creation of multivalent edges and prevent this (manifoldness)
        fast_collapse_bool = (self.manifold and (vl < NEIGHBORSIZE) and (vd < NEIGHBORSIZE))
        if fast_collapse_bool:
            # Do it the fast way if we can
            live_nn = self._vertices['neighbors'][_live_vertex]
            dead_nn =  self._vertices['neighbors'][_dead_vertex]
            live_mask = (live_nn != -1)
            dead_mask = (dead_nn != -1)
            live_list = self._halfedges['vertex'][live_nn[live_mask]]
            dead_list = self._halfedges['vertex'][dead_nn[dead_mask]]
        else:
            twin_mask = (self._halfedges['twin'] != -1)
            dead_mask = (self._halfedges['vertex'] == _dead_vertex)
            dead_list = self._halfedges['vertex'][self._halfedges['twin'][dead_mask & twin_mask]]
            live_list = self._halfedges['vertex'][self._halfedges['twin'][(self._halfedges['vertex'] == _live_vertex) & twin_mask]]
        twin_list = list((set(dead_list) & set(live_list)) - set([-1]))
        if len(twin_list) != 2:
            return
        
        if self.debug and (_live_vertex == _dead_vertex):
            print(_curr, curr_halfedge, _twin, twin_halfedge)
            raise RuntimeError('Live vertex equals dead vertex (both halves of edge point to same place)')

        # Collapse to the midpoint of the original edge vertices
        if fast_collapse_bool:
            self._halfedges['vertex'][self._halfedges['twin'][dead_nn[dead_mask]]] = _live_vertex
        else:
            self._halfedges['vertex'][dead_mask] = _live_vertex
        _live_pos = self._vertices['position'][_live_vertex]
        _dead_pos = self._vertices['position'][_dead_vertex]
        self._vertices['position'][_live_vertex] = 0.5*(_live_pos + _dead_pos)
        
        # update valence of vertex we keep
        self._vertices['valence'][_live_vertex] = vl + vd - 3
        
        if self.debug:
            print(self._vertices['valence'][_live_vertex], self._vertices['valence'][_dead_vertex])
            assert(self._vertices['valence'][_live_vertex] >=3)
        
        # delete dead vertex
        self._vertices[_dead_vertex] = -1

        # Zipper the remaining triangles
        def _zipper(edge1, edge2):
            # DB optimization notes:
            # 1) there is a (very?) small performance penalty for nested functions - move this up to class level?
            # 2) This would be a good candidate for a c/cython implementation (but possibly applies to whole edge collapse function)
            # 3) function can be simplified (see _zipper1 implementation below). Probably minimal performance impact in
            #    python version, but logic flow likely necessary in c/cython (avoids indexing with -1)
            t1 = self._halfedges['twin'][edge1]
            t2 = self._halfedges['twin'][edge2]

            if edge1 == -1:
                t1 = -1
            if edge2 == -1:
                t2 = -1
            
            if (t1 != -1) and (t2 != -1):
                self._halfedges['twin'][t2] = t1
                self._halfedges['twin'][t1] = t2
            else:
                if (t1 != -1):
                    self._halfedges['twin'][t1] = -1
                if (t2 != -1):
                    self._halfedges['twin'][t2] = -1

        def _zipper1(edge1, edge2):
            t1 = -1 if edge1 == -1 else self._halfedges['twin'][edge1]
            t2 = -1 if edge2 == -1 else self._halfedges['twin'][edge2]
    
            if (t1 != -1):
                self._halfedges['twin'][t1] = t2
                
            if (t2 != -1):
                self._halfedges['twin'][t2] = t1
            

        _zipper1(_next, _prev)
        if interior:
            _zipper1(_twin_next, _twin_prev)

        # We need some more pointers
        # TODO: make these safer
        _prev_twin = self._halfedges['twin'][_prev]
        _prev_twin_vertex = self._halfedges['vertex'][_prev_twin]
        _next_prev_twin = self._halfedges['next'][_prev_twin]
        _next_prev_twin_vertex = self._halfedges['vertex'][_next_prev_twin]
        if interior:
            _twin_next_vertex = self._halfedges['vertex'][_twin_next]
            _next_twin_twin_next = self._halfedges['next'][self._halfedges['twin'][_twin_next]]
            _next_twin_twin_next_vertex = self._halfedges['vertex'][_next_twin_twin_next]
            
        # Make sure we have good _vertex_halfedges references
        self._vertices['halfedge'][_live_vertex] = _prev_twin
        self._vertices['halfedge'][_prev_twin_vertex] = _next_prev_twin
        if interior:
            self._vertices['halfedge'][_twin_next_vertex] = self._halfedges['twin'][_twin_next]
            self._vertices['halfedge'][_next_twin_twin_next_vertex] = self._halfedges['next'][_next_twin_twin_next]

        # Grab faces to update
        face0 = self._halfedges['face'][self._halfedges['twin'][_next]]
        face1 = self._halfedges['face'][self._halfedges['twin'][_prev]]
        if interior:
            face2 = self._halfedges['face'][self._halfedges['twin'][_twin_next]]
            face3 = self._halfedges['face'][self._halfedges['twin'][_twin_prev]]

        # Delete the inner triangles
        self._face_delete(_curr)
        if interior:
            self._face_delete(_twin)

        if self.debug and interior:
            print(curr_halfedge, twin_halfedge)

        try:
            if live_update:
                if interior:
                    # Update faces
                    self._update_face_normals([face0, face1, face2, face3])
                    self._update_vertex_neighbors([_live_vertex, _prev_twin_vertex, _next_prev_twin_vertex, _twin_next_vertex])
                else:
                    self._update_face_normals([face0, face1])
                    self._update_vertex_neighbors([_live_vertex, _prev_twin_vertex, _next_prev_twin_vertex])
                self._faces_by_vertex = None
        
        except RuntimeError as e:
            print(_curr, _twin, _next, _prev, _twin_next, _twin_prev, _next_prev_twin, _next_twin_twin_next, _prev_twin)
            print([_live_vertex, _prev_twin_vertex, _next_prev_twin_vertex, _twin_next_vertex])
            raise e

    def _face_delete(self, _edge):
        """
        Delete a face defined by the halfedge _edge.

        Note that this does not account for adjusting halfedges, twins.

        Parameters
        ----------
            edge : int
                One self._halfedge index of the edge defining a face to delete.
        """
        if _edge == -1:
            return
        
        curr_edge = self._halfedges[_edge]
        if curr_edge['vertex'] == -1:
            return
        self._faces[curr_edge['face']] = -1

        self._face_vacancies.append(curr_edge['face'])
        
        self._edge_delete(curr_edge['next'])
        self._edge_delete(curr_edge['prev'])
        self._edge_delete(_edge)

    def _edge_delete(self, _edge):
        """
        Delete edge _edge in place from self._halfedges.

        Note that this does not account for adjusting halfedges, twins.

        Parameters
        ----------
            edge : int
                One self._halfedge index of the edge to delete.
        """
        if (_edge == -1):
            return
        
        self._halfedges[_edge] = -1
        self._halfedge_vacancies.append(_edge)

        self._faces_by_vertex = None

    def _insert(self, el, el_arr, el_vacancies, key, compact=False, insert_key=None, **kwargs):
        """
        Insert an element into an array at the position of the smallest empty 
        entry when searching the array by key.

        Parameters
        ----------
            el 
                Element to add to the array. Must be one of the dtypes in the 
                structured dtype of el_arr.
            el_arr : np.array
                Array where we add the element
            el_vacancies : list
                External list where we keep empty positions of el_arr.
            key : string
                Key on which to search the array
            insert_key : string
                Key on which to insert element el. This is by default the 
                search key.
            compact : bool
                Do we copy -1 values in the resize of el_arr?
            kwargs 
                List of dtype parameters to additionally define el.

        Returns
        -------
            ed
                Element added, includes el stored in full structured dtype of
                el_arr.
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
                # NOTE: If we search on self._vertices['position'] and one position is [-1,-1,-1],
                # which is not unreasonable, this will replace that vertex. Hence we provide the
                # option to search and insert on different keys.
                el_vacancies = [int(x) for x in np.argwhere(np.all(el_arr[key] == -1, axis=1))]
            else:
                # el_vacancies = [int(x) for x in np.flatnonzero(el_arr[key] == -1)]
                el_vacancies = np.flatnonzero(el_arr[key] == -1).tolist()

            idx = el_vacancies.pop(0)

        if idx == -1:
            raise ValueError('Index cannot be -1.')

        if insert_key is None:
            # Default to searching and inserting on the same key
            insert_key = key

        # Put el in el_arr
        ed = el_arr[idx]
        ed[insert_key] = el
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
                List of _halfedge dtype parameters (e.g. prev=10, next=2) to 
                define halfedge via more than just its vertex.

        Returns
        -------
            ed : HALFEDGE_DTYPE
                Halfedge created
            idx : int
                Index of the halfedge in self._halfedges.
        """

        ed, idx, self._halfedges, self._halfedge_vacancies = self._insert(vertex, self._halfedges, self._halfedge_vacancies, 'vertex', compact, None, **kwargs)
        
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
                List of _faces dtype parameters (e.g. area=13.556) to define 
                face via more than just one of its halfedges.

        Returns
        -------
            fa : FACE_DTYPE
                Face created
            idx : int
                Index of the face in self._faces.
        """

        fa, idx, self._faces, self._face_vacancies = self._insert(_edge, self._faces, self._face_vacancies, 'halfedge', compact, None, **kwargs)

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
                List of _vertices dtype parameters (e.g. valence=2) to define 
                vertex via more than just its position.

        Returns
        -------
            vx : VERTEX_DTYPE
                Vertex created
            idx : int
                Index of the vertex in self._vertices.
        """
        vx, idx, self._vertices, self._vertex_vacancies = self._insert(_vertex, self._vertices, self._vertex_vacancies, 'halfedge', compact, 'position', **kwargs)

        self._faces_by_vertex = None  # Reset

        return vx, idx

    def edge_split(self, _curr, live_update=True, upsample=False):
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
            upsample: bool
                Are we doing loop subdivision? If so, keep track of all edges
                incident on both a new vertex and an old verex that do not
                split an existing edge.
        """
        if _curr == -1:
            return
        
        curr_edge = self._halfedges[_curr]
        _prev = curr_edge['prev']
        _next = curr_edge['next']

        # Grab the new vertex position
        _vertex = 0.5*(self._vertices['position'][curr_edge['vertex'], :] + self._vertices['position'][self._halfedges['vertex'][_prev], :])
        _, _vertex_idx = self._new_vertex(_vertex)

        _twin = curr_edge['twin']
        interior = (_twin != -1)  # Are we on a boundary?
        
        if interior:
            twin_edge = self._halfedges[_twin]
            _twin_prev = twin_edge['prev']
            _twin_next = twin_edge['next']
        
        # Ensure the original faces have the correct pointers and add two new faces
        self._faces['halfedge'][curr_edge['face']] = _curr
        if interior:
            self._faces['halfedge'][twin_edge['face']] = _twin
            _, _face_1_idx = self._new_face(_twin_prev)
            self._halfedges['face'][_twin_prev] = _face_1_idx
        _, _face_2_idx = self._new_face(_next)
        self._halfedges['face'][_next] = _face_2_idx

        # Insert the new faces
        _, _he_0_idx = self._new_edge(self._halfedges['vertex'][_next], prev=_curr, next=_prev, face=self._halfedges['face'][_curr])
        if interior:
            _, _he_1_idx = self._new_edge(_vertex_idx, prev=_twin_next, next=_twin, face=self._halfedges['face'][_twin])
        
            _, _he_2_idx = self._new_edge(self._halfedges['vertex'][_twin_next], next=_twin_prev, face=_face_1_idx)
            _, _he_3_idx = self._new_edge(_vertex_idx, prev=_twin_prev, next=_he_2_idx, face=_face_1_idx)
            self._halfedges['prev'][_he_2_idx] = _he_3_idx

        _, _he_4_idx = self._new_edge(self._halfedges['vertex'][_curr], next=_next, face=_face_2_idx, twin=-1)
        _, _he_5_idx = self._new_edge(_vertex_idx, prev=_next, next=_he_4_idx, face=_face_2_idx)
        self._halfedges['prev'][_he_4_idx] = _he_5_idx

        self._halfedges['twin'][_he_0_idx] = _he_5_idx
        self._halfedges['twin'][_he_5_idx] = _he_0_idx

        if interior:
            self._halfedges['twin'][_he_1_idx] = _he_2_idx
            self._halfedges['twin'][_he_2_idx] = _he_1_idx

            self._halfedges['twin'][_he_3_idx] = _he_4_idx
            self._halfedges['twin'][_he_4_idx] = _he_3_idx

        # Update _prev, next
        self._halfedges['prev'][_prev] = _he_0_idx
        self._halfedges['prev'][_next] = _he_4_idx
        self._halfedges['next'][_next] = _he_5_idx

        if interior:
            # Update _twin_next, _twin_prev
            self._halfedges['next'][_twin_next] = _he_1_idx
            self._halfedges['prev'][_twin_prev] = _he_2_idx
            self._halfedges['next'][_twin_prev] = _he_3_idx

            self._halfedges['prev'][_twin] = _he_1_idx
        # Update _curr and _twin
        self._halfedges['vertex'][_curr] = _vertex_idx
        self._halfedges['next'][_curr] = _he_0_idx

        # Update halfedges
        if interior:
            self._vertices['halfedge'][self._halfedges['vertex'][_he_2_idx]] = _he_1_idx
        self._vertices['halfedge'][self._halfedges['vertex'][_prev]] = _curr
        self._vertices['halfedge'][self._halfedges['vertex'][_he_4_idx]] = _next
        self._vertices['halfedge'][_vertex_idx] = _he_4_idx
        self._vertices['halfedge'][self._halfedges['vertex'][_he_0_idx]] = _he_5_idx

        if upsample:
            # Make sure these edges emanate from the new vertex stored at _vertex_idx
            if interior:
                self._loop_subdivision_flip_edges.extend([_he_2_idx])
            
            self._loop_subdivision_flip_edges.extend([_he_0_idx])
            self._loop_subdivision_new_vertices.extend([_vertex_idx])

        if live_update:
            if interior:
                self._update_face_normals([self._halfedges['face'][_he_0_idx], self._halfedges['face'][_he_1_idx], self._halfedges['face'][_he_2_idx], self._halfedges['face'][_he_4_idx]])
                self._update_vertex_neighbors([self._halfedges['vertex'][_curr], self._halfedges['vertex'][_twin], self._halfedges['vertex'][_he_0_idx], self._halfedges['vertex'][_he_2_idx], self._halfedges['vertex'][_he_4_idx]])
            else:
                self._update_face_normals([self._halfedges['face'][_he_0_idx], self._halfedges['face'][_he_4_idx]])
                self._update_vertex_neighbors([self._halfedges['vertex'][_curr], self._halfedges['vertex'][_prev], self._halfedges['vertex'][_he_0_idx], self._halfedges['vertex'][_he_4_idx]])
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

        if (_curr == -1):
            return

        curr_edge = self._halfedges[_curr]
        _twin = curr_edge['twin']
        if (_twin == -1):
            # This is a boundary edge
            return
        _prev = curr_edge['prev']
        _next = curr_edge['next']
        
        twin_edge = self._halfedges[_twin]
        _twin_prev = twin_edge['prev']
        _twin_next = twin_edge['next']

        # Make sure both vertices have valence > 3 (preserve manifoldness)
        vc, vt = self._vertices['valence'][curr_edge['vertex']], self._vertices['valence'][twin_edge['vertex']]
        if (vc < 4) or (vt < 4):
            return

        # Calculate adjustments to the halfedges we're flipping
        new_v0 = self._halfedges['vertex'][_next]
        new_v1 = self._halfedges['vertex'][_twin_next]

        # If there's already an edge between these two vertices, don't flip (preserve manifoldness)
        # NOTE: This is potentially a problem if we start with a high-valence mesh. In that case, swap this
        # check with the more expensive commented one below.

        # Check for creation of multivalent edges and prevent this (manifoldness)
        fast_collapse_bool = (self.manifold and (vc < NEIGHBORSIZE) and (vt < NEIGHBORSIZE))
        if fast_collapse_bool:
            if new_v1 in self._vertices['neighbors'][new_v0]:
                return
        else:
            if new_v1 in self._halfedges['vertex'][self._halfedges['twin'][self._halfedges['vertex'] == new_v0]]:
                return

        # Convexity check: Let's see if the midpoint of the flipped edge will be above or below the plane of the 
        # current edge
        flip_midpoint = 0.5*(self._vertices['position'][new_v0] + self._vertices['position'][new_v1])
        plane_midpoint = (1./3)*(self._vertices['position'][curr_edge['vertex']] + self._vertices['position'][self._halfedges['vertex'][curr_edge['next']]] + self._vertices['position'][self._halfedges['vertex'][curr_edge['prev']]])
        flipped_dot = ((self._faces['normal'][curr_edge['face']])*(flip_midpoint - plane_midpoint)).sum()

        if flipped_dot < 0:
            # If flipping moves the midpoint of the edge below the original triangle's plane, this introduces
            # concavity, so don't flip.
            return

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
        Adjust vertices so they tend toward valence VALENCE 
        (or BOUNDARY_VALENCE for boundaries).
        """
        flip_count = 0

        # Do a single pass over all active edges and flip them if the flip minimizes the deviation of vertex valences
        # for i in np.arange(len(self._halfedges['vertex'])):
        #     if (self._halfedges['vertex'][i] < 0):
        #         continue
        for i in np.flatnonzero(self._halfedges['vertex'] != -1):

            curr_edge = self._halfedges[i]
            _twin = curr_edge['twin']
            twin_edge = self._halfedges[_twin]

            target_valence = VALENCE
            if _twin == -1:
                # boundary
                target_valence = BOUNDARY_VALENCE

            # Pre-flip vertices
            v1 = self._vertices['valence'][curr_edge['vertex']] - target_valence
            v2 = self._vertices['valence'][twin_edge['vertex']] - target_valence

            # Post-flip vertices
            v3 = self._vertices['valence'][self._halfedges['vertex'][curr_edge['next']]] - target_valence
            v4 = self._vertices['valence'][self._halfedges['vertex'][twin_edge['next']]] - target_valence

            # Check valence deviation from VALENCE (or
            # BOUNDARY_VALENCE for boundaries) pre- and post-flip
            score_pre = np.abs([v1,v2,v3,v4]).sum()
            score_post = np.abs([v1-1,v2-1,v3+1,v4+1]).sum()

            if score_post < score_pre:
                # Flip minimizes deviation of vertex valences from VALENCE (or
                # BOUNDARY_VALENCE for boundaries)
                self.edge_flip(i)
                flip_count += 1

        print('Flip count: %d' % (flip_count))

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

            # Weight by distance to neighors
            an = (1./self._halfedges['length'][nn])*nn_mask
            an[self._halfedges['length'][nn] == 0] = 0

            # Calculate gravity-weighted centroids
            A = np.sum(an, axis=1)
            c = 1./(A[...,None])*np.sum(an[...,None]*vn, axis=1)
            # Don't get messed up by slivers. NOTE: This is a hack in case
            # edge split/collapse don't do their jobs.
            A_mask = (A == 0)
            c[A_mask] = self._vertices['position'][A_mask]
            if self.fix_boundary:
                # Don't move vertices on a boundary
                boundary = self._halfedges[(self._halfedges['twin'] == -1)]
                tn = np.hstack([boundary['vertex'], self._halfedges['vertex'][boundary['prev']]])
                c[tn] = self._vertices['position'][tn]

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

        Follows procedure in Botsch and Kobbelt, A Remeshing Approach to 
        Multiresoluton Modeling, Eurographics Symposium on Geometry Processing,
        2004.

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

        if (target_edge_length == -1):
            # Guess edge_length
            target_edge_length = np.mean(self._halfedges['length'][self._halfedges['length'] != -1])

        for k in range(n):
            # 1. Split all edges longer than (4/3)*target_edge_length at their midpoint.
            split_count = 0
            for i in np.arange(len(self._halfedges['length'])):
                if (self._halfedges['vertex'][i] != -1) and (self._halfedges['length'][i] > 1.33*target_edge_length):
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
                if (self._halfedges['vertex'][i] != -1) and (self._halfedges['length'][i] < 0.8*target_edge_length):
                    self.edge_collapse(i)
                    collapse_count += 1
            print('Collapse count: ' + str(collapse_count))

            # 3. Flip edges in order to minimize deviation from VALENCE.
            self.regularize()

            # 4. Relocate vertices on the surface by tangential smoothing.
            self.relax(l=l, n=n_relax)

        # Let's double-check the mesh manifoldness
        self._manifold = None
        self.manifold

    def upsample(self, n=1):
        """
        Upsample the mesh by a factor of 4 (factor of 2 along each axis in the
        plane of the mesh) and smooth meshes by Loop's 5/8 and 3/8 rule.

        References:
            1. C. T. Loop, "Smooth Subdivision surfaces Based on Triangles," 
               University of Utah, 1987
            2. http://462cmu.github.io/asst2_meshedit/, task 4

        Parameters
        ----------
            n : int
                Number of upsampling operations to perform.
        """

        for k in range(n):
            # Reset
            split_edges = {}
            self._loop_subdivision_new_vertices = []
            self._loop_subdivision_flip_edges = []
            new_vertex_idxs = []

            # Mark the current edges as old
            edges_to_split = np.where(self._halfedges['length'] != -1)[0]
            split_halfedges = self._halfedges[edges_to_split]

            # Compute new vertex positions for old vertices
            old_vertex_idxs = split_halfedges['vertex']
            old_vertices = self._vertices[old_vertex_idxs]
            old_neighbors = old_vertices['neighbors']
            neg_mask = (old_neighbors != -1)
            # alpha is a function of vertex order N that satisifies constraints on equation 3.4 of Loop's thesis
            alpha = np.tanh(LOOP_ALPHA_FACTOR*old_vertices['valence'])
            old_vertex_positions = (alpha[:,None])*old_vertices['position'] + ((1-alpha)[:,None])*((1./old_vertices['valence'])[:,None])*np.sum(self._vertices['position'][self._halfedges['vertex'][old_neighbors]]*neg_mask[...,None],axis=1)
            
            # Compute new vertex positions for new vertices
            p0 = 0.375*(old_vertices['position'] + self._vertices['position'][self._halfedges['vertex'][split_halfedges['twin']]])
            p1 = 0.125*(self._vertices['position'][self._halfedges['vertex'][split_halfedges['next']]] + self._vertices['position'][self._halfedges['vertex'][self._halfedges['next'][split_halfedges['twin']]]])
            new_vertex_positions = p0 + p1

            # 1. Split every edge in the mesh
            for j, i in enumerate(edges_to_split):
                if i in split_edges:
                    continue
                
                split_edges[self._halfedges['twin'][i]] = i
                new_vertex_idxs.append(j)
                self.edge_split(i, upsample=True)
            
            # 2. Flip any new edge that touches an old vertex and a new vertex
            edges_to_flip = list(set(self._halfedges['vertex'][self._loop_subdivision_flip_edges]) - set(self._loop_subdivision_new_vertices))
            for e in edges_to_flip:
                self.edge_flip(e)

            # Get any boundary vertices
            boundary_vertex_idxs = self._halfedges['vertex'][(self._halfedges['twin'] == -1)]
            boundary_vertex_positions = self._vertices['position'][boundary_vertex_idxs]

            # Copy new position values into self._vertices['position']
            self._vertices['position'][old_vertex_idxs] = old_vertex_positions
            self._vertices['position'][self._loop_subdivision_new_vertices] = new_vertex_positions[new_vertex_idxs]

            # Restore boundary vertex positions
            self._vertices['position'][boundary_vertex_idxs] = boundary_vertex_positions

        self._manifold = None

    def downsample(self, n_triangles=None):
        """
        Mesh downsampling via quadratic error metrics.

        Follows procedure in Garland and Heckbert, Surface Simplification Using
        Quadtratic Error Metrics.

        Parameters
        ----------
            n_triangles : int
                Target number of triangles in the downsampled mesh.
        """

        import PYME.experimental._treap as treap

        # 1. Compute quadratic matrices for all initial vertices
        v = self._vertices['position']
        n = self._vertices['normal']
        d = (-n*v).sum(1)
        p = np.vstack([n.T, d]).T  # plane
        vn = self._vertices['neighbors']
        vn_mask = (vn != -1)
        pp = p[self._halfedges['vertex'][vn]]*vn_mask[...,None]  # neighbor planes
        Q = (pp[...,None]*pp[:,:,None,:]).sum(1)  # per-vertex quadratic

        # 2. Select all valid pairs
        #   For now, operate on edges. In the future we can also operate on any
        #   pair of vertex positions such that || v_1 - v_2 || < t, where v_1,
        #   v_2 \in R^3, and t \in R is a threshold.

        # 3. Compute optimal contraction target and cost of each contraction
        # NOTE: we do this for all halfedges, empty or not. This saves us from having to search for
        # unique halfedges later in step 5. Instead, we just ignore -1 values.
        edges_mask = (self._halfedges['vertex'] != -1)
        edges = np.vstack([self._halfedges['vertex'], self._halfedges['vertex'][self._halfedges['prev']]]).T
        # Mask and shift to avoid a copy operation on Q
        Q_mask = np.ones((4,4))
        Q_mask[3,:] = 0
        Q_shift = np.zeros((4,4))
        Q_shift[3,3] = 1
        # h is the solution vector for least squares
        h = np.array([0,0,0,1])
        # 3a. Compute the optimal contraction target for each valid pair
        Q_edge = (Q[edges]).sum(1)  # Q paired
        try:
            vp = (np.linalg.inv(Q_edge*Q_mask[None,...] + Q_shift[None,...])*h).sum(2)
        except(np.linalg.LinAlgError):
            vp = 0.5*np.sum(self._vertices['position'][edges],axis=1)
            vp = np.vstack([vp.T, np.ones(vp.shape[0])]).T

        # 3b. Compute cost of contractions
        err = ((vp[:,None,:]*Q_edge).sum(2)*vp).sum(1)

        # 4. Place all the pairs in a heap keyed on cost with the minimum cost pair on top
        cost_heap = treap.Treap()
        for i in (np.arange(len(edges))[edges_mask]):
            cost_heap.insert(np.float32(err[i]), np.int32(i))
            
        # 5. Iteratively remove the pair of least cost from the heap, 
        #    contract this pair, and update the costs of all valid 
        #    pairs involving the remaining live vertex.
        n_faces = np.sum(self._faces['halfedge'] != -1)
        if n_triangles is None:
            # Assume we want to downsample by a factor of 4
            n_triangles = int(0.25*n_faces)
        collapse_tries = int(0.125*(n_faces - n_triangles))
        while True:
            if n_faces <= n_triangles:
                # Sometimes edge_collapse does not collapse the edge
                # because it would affect the manifoldness of the mesh,
                # so we need to see how far off we are from the desired
                # number of triangles n_triangles.
                tmp = np.sum(self._faces['halfedge'] != -1)
                if (n_faces == tmp) or (collapse_tries == 0):
                    # We don't want to double check ourselves forever
                    break
                # Update our estimate and keep refining the mesh
                n_faces = tmp
                collapse_tries -= 1
            
            # Grab the edge with minimum cost
            _, _edge = cost_heap.pop()

            if _edge == -1:
                print('Edge was -1???')
                continue

            # Compute the new quadratic
            edge = self._halfedges[_edge]
            # Grab the live and dead vertex
            _live_vertex = edge['vertex']
            _dead_vertex = self._halfedges['vertex'][edge['prev']]
            Q_edge[_edge] = Q[_live_vertex] + Q[_dead_vertex]

            if (_live_vertex == -1) or (_dead_vertex == -1):
                print('Vertex was -1???')
                continue

            # Grab all neighboring halfedges
            nn_live = self._vertices['neighbors'][_live_vertex]
            nn_live_mask = (nn_live != -1)
            nn_live_twin = self._halfedges['twin'][nn_live[nn_live_mask]]
            nn_live_twin_mask = (nn_live_twin != -1)
            nn_dead = self._vertices['neighbors'][_dead_vertex]
            nn_dead_mask = (nn_dead != -1)
            nn_dead_twin = self._halfedges['twin'][nn_dead[nn_dead_mask]]
            nn_dead_twin_mask = (nn_dead_twin != -1)
            neighbors = np.hstack([nn_live[nn_live_mask], nn_live_twin[nn_live_twin_mask], nn_dead[nn_dead_mask], nn_dead_twin[nn_dead_twin_mask]])

            # Eliminate edges touching either vertex from the cost heap
            for neighbor in neighbors:
                cost_heap.delete(np.int32(neighbor))
                            
            # Collapse the edge
            self.edge_collapse(_edge)
            
            # Decrement the estimate of the number of triangles in the 
            # mesh accordingly
            n_faces -= 2

            # Set the quadratic of _live_vertex to the edge quadratic
            Q[_live_vertex] = Q_edge[_edge] 
            # self._vertices['position'][_live_vertex] = vp[_edge,:3]

            # Grab the new neighbors
            nn_live = self._vertices['neighbors'][_live_vertex]
            nn_live_mask = (nn_live != -1)
            nn_live_twin = self._halfedges['twin'][nn_live[nn_live_mask]]
            nn_live_twin_mask = (nn_live_twin != -1)
            neighbors = np.hstack([nn_live[nn_live_mask], nn_live_twin[nn_live_twin_mask]])

            # Update the costs and positions of each of these neighbors and re-insert them into the queue
            new_edges = np.vstack([self._halfedges['vertex'][neighbors], self._halfedges['vertex'][self._halfedges['prev'][neighbors]]]).T
            Q_edge[neighbors] = (Q[new_edges]).sum(1)

            try:
                vp[neighbors] = (np.linalg.inv(Q_edge[neighbors]*Q_mask[None,...] + Q_shift[None,...])*h).sum(2)
            except(np.linalg.LinAlgError):
                vp_temp = 0.5*np.sum(self._vertices['position'][new_edges],axis=1)
                vp[neighbors] = np.vstack([vp_temp.T, np.ones(vp_temp.shape[0])]).T

            err[neighbors] = ((vp[neighbors,None,:]*Q_edge[neighbors]).sum(2)*vp[neighbors]).sum(1)

            for n in neighbors:
                cost_heap.insert(np.float32(err[n]), np.int32(n))

        self._manifold = None

    def find_connected_components(self):
        """
        Label connected components of the mesh.
        """ 

        # Set all components to -1 (background)
        self._halfedges['component'] = -1
        self._vertices['component'] = -1
        self._faces['component'] = -1

        # Connect component by vertex
        component = 0
        for _vertex in np.arange(self._vertices.shape[0]):
            curr_vertex = self._vertices[_vertex]
            if (curr_vertex['halfedge'] == -1) or (curr_vertex['component'] != -1):
                # We only want to deal with valid, unassigned vertices
                continue

            # Search vertices contains vertices connected to this vertex.
            # We want to see if there are other vertices connected to the
            # search vertices.
            search_vertices = []
            self._vertices[_vertex]['component'] = component
            search_vertices.append(_vertex)

            # Find vertices connected to the search vertices
            while search_vertices:
                # Grab the latest search vertex neighbors
                _v = search_vertices.pop()
                nn = self._vertices['neighbors'][_v]
                # Loop over the neighbors
                for _edge in nn:
                    if _edge == -1:
                        continue
                    # Assign the vertex component label to halfedges and faces
                    # emanating from this vertex
                    curr_edge = self._halfedges[_edge]
                    self._halfedges['component'][_edge] = component
                    self._halfedges['component'][curr_edge['prev']] = component
                    self._halfedges['component'][curr_edge['next']] = component
                    self._faces['component'][curr_edge['face']] = component
                    
                    # If the vertex is assigned, we've already visited it and
                    # don't need to do so again.
                    if self._vertices['component'][curr_edge['vertex']] != -1:
                        continue
                    self._vertices['component'][curr_edge['vertex']] = component
                    search_vertices.append(curr_edge['vertex'])
            
            # Increment the label as any vertices not included in this
            # iteration's search will be part of another component.
            component += 1

    def keep_largest_connected_component(self):
        # Find the connected components
        self.find_connected_components()

        # Which connected component is largest? 
        com, counts = np.unique(self._vertices['component'][self._vertices['component']!=-1], return_counts=True)
        max_count = np.argmax(counts)
        max_com = com[max_count]

        # Remove the smaller components
        _vertices = np.where((self._vertices['component'] != max_com))[0]
        _edges = np.where((self._halfedges['component'] != max_com))[0]
        _edges_with_twins = _edges[self._halfedges['twin'][_edges] != -1]
        _faces = np.where((self._faces['component'] != max_com))[0]
        self._vertices[_vertices] = -1
        _kept_edges = self._halfedges['twin'][_edges_with_twins]
        self._halfedges['twin'][_kept_edges] = -1
        self._halfedges[_edges] = -1
        self._faces[_faces] = -1

        # Mark deleted faces as available
        self._vertex_vacancies = list(set(self._vertex_vacancies).union(set(_vertices)))
        self._halfedge_vacancies = list(set(self._halfedge_vacancies).union(set(_edges)))
        self._face_vacancies = list(set(self._face_vacancies).union(set(_faces)))

        # Update newly orphaned edges
        self._update_face_normals(list(set(self._halfedges['face'][_kept_edges])))
        self._update_vertex_neighbors(list(set(self._halfedges['vertex'][_kept_edges])))
        
        self._faces_by_vertex = None

    def _find_boundary_polygons(self):
        """
        Return a list of closed polygons, each defined by a list of halfedges,
        where each polygon defines a boundary in the mesh.
        """
        boundary_polygons = []

        # 1. Construct an initial list of boundary edges
        boundary_edges = list(np.where((self._halfedges['twin'] == -1) & (self._halfedges['vertex'] != -1))[0])

        # 2. Get the boundary polygons
        while len(boundary_edges) > 0:
            _root = boundary_edges[-1]  # initial halfedge defining the polygon
            curr_polygon = [_root]
            # Step next-twin-next to move around the boundary
            _next = self._halfedges['next'][_root]
            _twin = self._halfedges['twin'][_next]
            _curr = self._halfedges['next'][_twin]
            if self._halfedges['twin'][_next] == -1:
                _curr = _next
            if self._halfedges['twin'][_twin] == -1:
                _curr = _twin
            
            # We check open_chain instead of _curr != root since open_chain accounts for closed polygons
            # ending at an isolated singular vertex (two boundaries emit from a single vertex)
            open_chain = (self._halfedges['vertex'][self._halfedges['prev'][_root]] != self._halfedges['vertex'][curr_polygon[-1]])
            
            # Loop around the boundary until we return the initial halfedge
            max_polygon_size = 100  # Put an upper limit on the polygon sizes so we don't look forever
            while open_chain and (max_polygon_size > 0):
                # We're not always guaranteed to hit the boundary again right 
                # away doing next-twin-next, so wait until we do
                twin_next_attemps = 2*BOUNDARY_VALENCE
                while (twin_next_attemps > 0) and (self._halfedges['twin'][_curr] != -1):
                    
                    # Need to do twin next in between boundary edges
                    _twin = self._halfedges['twin'][_curr]
                    _curr = self._halfedges['next'][_twin]
                    if _curr == -1:
                        # We hit a isolated singular vertex
                        break
                    twin_next_attemps -= 1
                
                if _curr == -1:
                    break
                
                curr_polygon.append(_curr)

                _next = self._halfedges['next'][_curr]
                _twin = self._halfedges['twin'][_next]
                _curr = self._halfedges['next'][_twin]

                if self._halfedges['twin'][_next] == -1:
                    _curr = _next
                if self._halfedges['twin'][_twin] == -1:
                    _curr = _twin

                open_chain = (self._halfedges['vertex'][self._halfedges['prev'][_root]] != self._halfedges['vertex'][curr_polygon[-1]])
                max_polygon_size -= 1

            if max_polygon_size == 0:
                print('Polygon search took too long. Canceled.')

            if ((_curr == -1) and (open_chain)) or (max_polygon_size == 0):
                # We failed to find a boundary polygon
                boundary_edges.pop()
                continue
            else:
                # We found a boundary polygon!
                boundary_polygons.append(curr_polygon)
                # Delete the polygon from the potential polygon list
                for edge in curr_polygon:
                    if edge in boundary_edges:
                        boundary_edges.remove(edge)

        return boundary_polygons

    def _fill_triangle(self, h0, h1, h2):
        """
        Create a triangle inside three halfedges.

        Note that this does not update the vertices or faces.
        """

        _h0_twin_vertex = self._halfedges['vertex'][h2]
        _, _h0_twin = self._new_edge(_h0_twin_vertex, twin=h0)
        _, _face = self._new_face(_h0_twin)
        _h1_twin_vertex = self._halfedges['vertex'][h0]
        _, _h1_twin = self._new_edge(_h1_twin_vertex, twin=h1, face=_face, next=_h0_twin)
        self._halfedges['face'][_h0_twin] = _face
        self._halfedges['prev'][_h0_twin] = _h1_twin
        _h2_twin_vertex = self._halfedges['vertex'][h1]
        _, _h2_twin = self._new_edge(_h2_twin_vertex, twin=h2, face=_face, prev=_h0_twin, next=_h1_twin)
        self._halfedges['next'][_h0_twin] = _h2_twin
        self._halfedges['prev'][_h1_twin] = _h2_twin

        if h0 != -1:
            self._halfedges['twin'][h0] = _h0_twin
        if h1 != -1:
            self._halfedges['twin'][h1] = _h1_twin
        if h2 != -1:
            self._halfedges['twin'][h2] = _h2_twin

    def _fan_triangulation(self, polygon):
        """
        Triangulate a polygon, defined by halfedges in the mesh,
        by creating a fan emanating from a single vertex out to
        the boundary halfedges.

        Parameters
        ----------
            polygon : list
                List of halfedges defining a closed polygon in the mesh.
        """

        if len(polygon) < 3:
            raise ValueError('polygon must be of length >= 3')

        # Reverse order
        polygon = polygon[::-1]

        while len(polygon) > 0:
            h0 = polygon.pop()
            h1 = polygon.pop()
            if (len(polygon) == 1):
                # Base case, make a triangle
                # polygon = polygon[::-1]
                h2 = polygon.pop()
                self._fill_triangle(h0,h1,h2)
                _h0_twin = self._halfedges['twin'][h0]
            else:
                # Create a new triangle with a boundary edge
                self._fill_triangle(h0, h1, -1)
                
                # Due to the way fill_triangle works, we need to set h0's vertex after the fact
                _h0_twin = self._halfedges['twin'][h0]
                self._halfedges['vertex'][_h0_twin] = self._halfedges['vertex'][self._halfedges['prev'][h0]]
                
                # Adjust the boundary
                polygon.append(self._halfedges['next'][_h0_twin])

            # Update faces and vertices
            self._update_face_normals([self._halfedges['face'][_h0_twin], self._halfedges['face'][h0], self._halfedges['face'][h1]])
            self._update_vertex_neighbors([self._halfedges['vertex'][_h0_twin], self._halfedges['vertex'][h0], self._halfedges['vertex'][h1]])

    def _zig_zag_triangulation(self, polygon):
        """
        Triangulate a polygon, defined by halfedges in the mesh,
        by creating zig-zagging between edges of the boundary.

        Parameters
        ----------
            polygon : list
                List of halfedges defining a closed polygon in the mesh.
        """

        if len(polygon) < 3:
            raise ValueError('polygon must be of length >= 3')

        # Reverse order
        polygon = polygon[::-1]
        odd = True

        while len(polygon) > 0:
            h0 = polygon.pop()
            h1 = polygon.pop()
            if (len(polygon) == 1):
                # Base case, make a triangle
                # polygon = polygon[::-1]
                h2 = polygon.pop()
                self._fill_triangle(h0,h1,h2)
                _h0_twin = self._halfedges['twin'][h0]
            else:
                # Create a new triangle with a boundary edge
                self._fill_triangle(h0, h1, -1)
                
                # Due to the way fill_triangle works, we need to set h0's vertex after the fact
                _h0_twin = self._halfedges['twin'][h0]
                self._halfedges['vertex'][_h0_twin] = self._halfedges['vertex'][self._halfedges['prev'][h0]]
            
                # Adjust the boundary
                polygon = polygon[::-1]  # zig-zag
                if odd:
                    polygon.insert(-1,self._halfedges['next'][_h0_twin])
                else:
                    polygon.append(self._halfedges['next'][_h0_twin])
                odd = (not odd)

            # Update faces and vertices
            self._update_face_normals([self._halfedges['face'][_h0_twin], self._halfedges['face'][h0], self._halfedges['face'][h1]])
            self._update_vertex_neighbors([self._halfedges['vertex'][_h0_twin], self._halfedges['vertex'][h0], self._halfedges['vertex'][h1]])

    def _fill_holes(self, method='zig-zag'):
        """
        Fill holes in the mesh.

        Parameters
        ----------
            method : string
                The method to use to patch boundary polygons. A string listed 
                in options. 
        """

        options = ['fan', 'zig-zag']

        if method not in options:
            print('Unknown triangulation method. Using default.')
            method = 'zig-zag'

        boundary_polygons = self._find_boundary_polygons()
        
        # Find length 2 polygons and zipper them ("pinch")
        to_remove = []
        for polygon in boundary_polygons:
            if len(polygon) == 2:
                self._halfedges['twin'][polygon[0]] = polygon[1]
                self._halfedges['twin'][polygon[1]] = polygon[0]
                to_remove.append(polygon)
        
        # Eliminate the length 2 polygons before triangulation
        for polygon in to_remove:
            boundary_polygons.remove(polygon)

        # Triangulate
        if method == 'fan':
            for polygon in boundary_polygons:
                self._fan_triangulation(polygon)
        
        if method == 'zig-zag':
            for polygon in boundary_polygons:
                self._zig_zag_triangulation(polygon)

        # Reset display
        self._faces_by_vertex = None

    def _connect_neighbors(self, _nn, starting_component=0):
        """
        Connect the neighbors in the list _nn starting from
        starting_component.

        Parameters
        ----------
            _nn : list
                List of neighbors to connect.
            starting_component : int
                Component numbering to start at.

        Returns
        -------
            component : int
                Maxmimum component number used to group neighbors.
        """
        # Assign components via the 1-neighbors
        component = starting_component
        for _neighbor in _nn:
            singular = (_neighbor in self.singular_edges)
            if singular:
                component += 1
            if self._faces['component'][self._halfedges['face'][_neighbor]] != -1:
                continue
            self._faces['component'][self._halfedges['face'][_neighbor]] = component

            # Extra check in case we're examining the edges out of order
            if singular:
                component += 1

        # Loop through the neighbors again and group connected components
        for _neighbor in _nn:
            if (_neighbor in self.singular_edges):
                continue
    
            _twin_neighbor = self._halfedges['twin'][_neighbor]
            if _twin_neighbor != -1:
                _face = self._halfedges['face'][_neighbor]
                _twin_face = self._halfedges['face'][_twin_neighbor]
                min_component = np.min([self._faces['component'][_face], self._faces['component'][_twin_face]])
                self._faces['component'][_face] = min_component
                self._faces['component'][_twin_face] = min_component

        return component

    def _group_neighbors(self, _vertex):
        """
        Group faces around the 1-neighbor ring of a vertex by reachability. 
        Faces are not reachable if they are separated by a singular edge.

        Parameters
        ----------
            _vertex : int
                Index of vertex at the center of a 1-neighbor ring.
        """

        # Zero out connectivity
        self._vertices['component'][:] = -1
        self._faces['component'][:] = -1
        self._halfedges['component'][:] = -1

        # Grab the neighbors
        _nn = self._vertices[_vertex]['neighbors']
        _nn = _nn[_nn!=-1]

        # Connect the neighbors
        component = self._connect_neighbors(_nn, 0)

        # Don't trust self._vertices['neighbors'] because we're dealing with 
        # singular vertices
        _twin_nn = list(np.where(self._halfedges['vertex'] == _vertex)[0])

        # If there were any incident vertices not included in this list, group them, too
        _remaining = list(set(_twin_nn) - set(self._halfedges['twin'][_nn]) - set([-1]))
        
        if _remaining:
            component += 1  # We're off the original 1-neighbor ring, so disconnect
            self._connect_neighbors(_remaining, component)

    def _remove_singularities(self):
        """
        Excise singularities from the mesh. This essentially unzips multivalent
        edges from the rest of the mesh, creating new, smaller connected 
        components, each of which is free of singularities. This follows the
        local method for cutting outlined in Gueziec et al.

        References
        ----------
            Gueziec et al. Cutting and Stitching: Converting Sets of Polygons
            to Manifold Surfaces, IEEE TRANSACTIONS ON VISUALIZATION AND 
            COMPUTER GRAPHICS, 2001.
        """

        # Make sure we're working with the latest singular edges/vertices
        self._singular_edges = None
        self._singular_vertices = None

        #  For each marked vertex, partition the faces of the 1-neighbor ring
        #    into nc "is reachable" equivalence classes. Faces are reachable if
        #    they share a non-singular edge incident on the marked vertex.
        for _vertex in self.singular_vertices[::-1]:

            _twin_nn = list(np.where(self._halfedges['vertex'] == _vertex)[0])
            _update_vertices = np.hstack([_vertex, self._halfedges['vertex'][self._halfedges['twin'][_twin_nn]]])
            # _update_faces = self._faces[self._halfedges['face'][_twin_nn]]

            self._group_neighbors(_vertex)
            
            # Create nc-1 copies of the vertex and assign each equivalence class 
            #    one of these vertices. (All but one component gets a new vertex).
            components = np.unique(self._faces['component'][self._faces['component'] != -1])
            for c in components[1:]:
                _faces = self._faces[self._faces['component'] == c]
                _edges = np.hstack([self._halfedges['prev'][_faces['halfedge']], _faces['halfedge'], self._halfedges['next'][_faces['halfedge']]])
                _vertices = self._halfedges['vertex'][_edges]
                
                _modified_edges = _edges[_vertices == _vertex] 

                _edge = self._halfedges['next'][_modified_edges[0]]

                _, _new_vertex = self._new_vertex(self._vertices['position'][_vertex], halfedge=_edge)
                # Assign edges in this component connected to _vertex to _new_vertex
                self._halfedges['vertex'][_modified_edges] = _new_vertex

                # Disconnect the boundaries of the components
                # Any edge pointing to the new vertex with a twin in a 
                # different component must be reassigned
                _twin_face_component = self._faces['component'][self._halfedges['face'][self._halfedges['twin'][_modified_edges]]]
                _boundary = (_twin_face_component != c)
                _twin_edges = self._halfedges['twin'][_modified_edges[_boundary]]
                _twin_edges_mask = (_twin_edges != -1)
                _valid_edges = _modified_edges[_boundary][_twin_edges_mask]
                self._halfedges['twin'][_twin_edges[_twin_edges_mask]] = -1
                self._halfedges['twin'][_valid_edges] = -1
                # Any edge emanating from the new vertex with a twin in a
                # different component must be reassigned
                _twin_next_face_component = self._faces['component'][self._halfedges['face'][self._halfedges['twin'][self._halfedges['next'][_modified_edges]]]]
                _boundary = (_twin_next_face_component != c)                
                _twin_next_edges = self._halfedges['twin'][self._halfedges['next'][_modified_edges[_boundary]]]
                _twin_next_edges_mask = (_twin_next_edges != -1)
                _valid_edges = self._halfedges['next'][_modified_edges[_boundary][_twin_next_edges_mask]]
                self._halfedges['twin'][_twin_next_edges[_twin_next_edges_mask]] = -1
                self._halfedges['twin'][_valid_edges] = -1

                # Double check that the original vertex did not have a halfedge 
                # in this component
                _safe_faces_bool = (self._faces['component'][self._halfedges['face']] != c)
                _safe_vertices = np.where((self._halfedges['vertex'] == _vertex) & (_safe_faces_bool))[0]
                _safe_halfedge = self._halfedges['next'][_safe_vertices[0]]
                self._vertices['halfedge'][_vertex] = _safe_halfedge

                # self._update_face_normals(_update_faces)
                self._update_vertex_neighbors(np.hstack([_update_vertices, _new_vertex]))

        # Reset
        self._singular_edges = None
        self._singular_vertices = None

    def repair(self):
        """
        Repair the mesh so it's topologically manifold.

        References
        ----------
            M. Attene, A lightweight approach to repairing digitized polygon
            meshes, The Visual Computer, 2010.
        """
        
        # 1. Remove singularities (singular edges and isolated singular vertices)
        self._remove_singularities()

        # 2. Remove all connected components except the largest
        self.keep_largest_connected_component()

        # 3. Patch mesh holes with new triangles
        self._fill_holes()

        self._manifold = None

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

    def to_ply(self, filename, colors=None):
        """
        Save a list of triangles and their vertex colors to a PLY file.
        We assume colors are a Nx3 array where N is the number of vertices.
        """
        from PYME.IO.FileUtils import ply

        # Construct a re-indexing for non-negative vertices
        live_vertices = np.flatnonzero(self._vertices['halfedge'] != -1)
        new_vertex_indices = np.arange(live_vertices.shape[0])
        vertex_lookup = np.zeros(self._vertices.shape[0], dtype=int)
        
        vertex_lookup[live_vertices] = new_vertex_indices

        # Grab the faces and vertices we want
        faces = vertex_lookup[self.faces]
        vertices = self._vertices['position'][live_vertices]

        ply.save_ply(filename, vertices, faces, colors)
