# -*- coding: utf8 -*-
# cython: profile=True
cimport numpy as np
import numpy as np
cimport cython

import warnings

#from libc.math cimport sqrtf

cdef extern from "<math.h>" nogil:
    float sqrtf(float)

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

import copy

from PYME.experimental import triangle_mesh_utils

DEF MAX_VERTEX_COUNT = 2**31

# Regular, boundary, and max vertex valences. These need tp be
DEF VALENCE = 6
DEF BOUNDARY_VALENCE = 4
    
cdef extern from "triangle_mesh_utils.c":
    void _update_face_normals(np.int32_t *f_idxs, halfedge_t *halfedges, vertex_t *vertices, face_t *faces, signed int n_idxs)
    
    void update_face_normal(int f_idx, halfedge_t *halfedges, vertex_d *vertices, face_d *faces)
    void update_single_vertex_neighbours(int v_idx, halfedge_t *halfedges, vertex_d *vertices, face_d *faces)

    int flood_fill_star_component(np.int32_t h_idx, int component, halfedge_t *halfedges)
    float norm(const float *pos)
    void cross(const float *a, const float *b, float *n)
    void difference(const float *a, const float *b, float *out)
    void vsum(const float *a, const float *b, float *out)
    float dot(const float *a, const float *b)
    void scalar_mult(float *a, const float b);

cdef extern from "remesh_ops.c":
    int remesh_edge_flip(halfedge_t * halfedges, vertex_t *vertices, face_t * faces, int idx, int n_halfedges, int live_update);
    int remesh_relax(halfedge_t *halfedges, vertex_t *vertices, int n_vertices, face_t* faces, int n_faces, float l, int n_iterations);
    void remesh_edge_zipper(halfedge_t * halfedges, np.int32_t edge1, np.int32_t edge2);
    int remesh_check_neighour_twins(halfedge_t * halfedges, vertex_t * vertices, np.int32_t vertex_id);
    int remesh_edge_collapse(halfedge_t * halfedges, np.int32_t n_halfedges, vertex_t * vertices, face_t* faces, np.int32_t idx, int live_update);
    int remesh_delete_vertex(vertex_t * vertices, np.int32_t v_idx);
    int remesh_delete_edge(halfedge_t * halfedges, np.int32_t e_idx);
    int remesh_delete_face(halfedge_t * halfedges, face_t * faces, np.int32_t e_idx);
    int remesh_split_edge(halfedge_t * halfedges, np.int32_t n_halfedges, vertex_t * vertices, face_t* faces, np.int32_t idx, np.int32_t * new_edges,
                      np.int32_t *new_vertices, np.int32_t * new_faces, int n_edge_idx, int n_vertex_idx, int n_face_idx, int live_update, int upsample)


HALFEDGE_DTYPE = np.dtype([('vertex', 'i4'), ('face', 'i4'), ('twin', 'i4'), ('next', 'i4'), ('prev', 'i4'), ('length', 'f4'), ('component', 'i4'), ('locally_manifold', 'i4')], align=True)
FACE_DTYPE = np.dtype([('halfedge', 'i4'), ('normal', '3f4'), ('area', 'f4'), ('component', 'i4')], align=True)
FACE_DTYPE2 = np.dtype([('halfedge', 'i4'), ('normal0', 'f4'), ('normal1', 'f4'), ('normal2', 'f4'), ('area', 'f4'), ('component', 'i4')], align=True)
# Note that VERTEX_DTYPE neighbors size must match NEIGHBORSIZE
VERTEX_DTYPE = np.dtype([('position', '3f4'), ('normal', '3f4'), ('halfedge', 'i4'), ('valence', 'i4'), ('neighbors', '20i4'), ('component', 'i4'),('locally_manifold', 'i4')], align=True)
VERTEX_DTYPE2 = np.dtype([('position0', 'f4'), 
                          ('position1', 'f4'), 
                          ('position2', 'f4'), 
                          ('normal0', 'f4'), 
                          ('normal1', 'f4'), 
                          ('normal2', 'f4'), 
                          ('halfedge', 'i4'), 
                          ('valence', 'i4'), 
                          ('neighbor0', 'i4'),
                          ('neighbor1', 'i4'), 
                          ('neighbor2', 'i4'), 
                          ('neighbor3', 'i4'), 
                          ('neighbor4', 'i4'), 
                          ('neighbor5', 'i4'), 
                          ('neighbor6', 'i4'), 
                          ('neighbor7', 'i4'), 
                          ('neighbor8', 'i4'), 
                          ('neighbor9', 'i4'), 
                          ('neighbor10', 'i4'), 
                          ('neighbor11', 'i4'), 
                          ('neighbor12', 'i4'), 
                          ('neighbor13', 'i4'), 
                          ('neighbor14', 'i4'), 
                          ('neighbor15', 'i4'), 
                          ('neighbor16', 'i4'), 
                          ('neighbor17', 'i4'), 
                          ('neighbor18', 'i4'), 
                          ('neighbor19', 'i4'), 
                          ('component', 'i4'),
                          ('locally_manifold', 'i4')], align=True)


# MESH CONVENTIONS
##################
#
# halfedge.vertex = vertex the halfedge is pointing TO
#
# vertex.halfedge = a halfedge that eminates FROM the vertex.
#

# cdef packed struct halfedge_d:
#     np.int32_t vertex
#     np.int32_t face
#     np.int32_t twin
#     np.int32_t next
#     np.int32_t prev
#     np.float32_t length
#     np.int32_t component
#
# cdef packed struct face_d:
#     np.int32_t halfedge
#     np.float32_t normal0
#     np.float32_t normal1
#     np.float32_t normal2
#     np.float32_t area
#     np.int32_t component
    
# cdef packed struct face_t: #non flattened type from triangle_mesh_utils.h
#     np.int32_t halfedge
#     float normal[VECTORSIZE]
#     float area
#     np.int32_t component

# cdef packed struct vertex_d:
#     np.float32_t position0
#     np.float32_t position1
#     np.float32_t position2
#     np.float32_t normal0
#     np.float32_t normal1
#     np.float32_t normal2
#     np.int32_t halfedge
#     np.int32_t valence
#     np.int32_t neighbor0
#     np.int32_t neighbor1
#     np.int32_t neighbor2
#     np.int32_t neighbor3
#     np.int32_t neighbor4
#     np.int32_t neighbor5
#     np.int32_t neighbor6
#     np.int32_t neighbor7
#     np.int32_t neighbor8
#     np.int32_t neighbor9
#     np.int32_t neighbor10
#     np.int32_t neighbor11
#     np.int32_t neighbor12
#     np.int32_t neighbor13
#     np.int32_t neighbor14
#     np.int32_t neighbor15
#     np.int32_t neighbor16
#     np.int32_t neighbor17
#     np.int32_t neighbor18
#     np.int32_t neighbor19
#     np.int32_t component

# cdef packed struct vertex_t:  # non-flattened version from triangle_mesh_utils.c
#     float position[VECTORSIZE];
#     float normal[VECTORSIZE];
#     np.int32_t halfedge;
#     np.int32_t valence;
#     np.int32_t neighbors[NEIGHBORSIZE];
#     np.int32_t component;

#cdef union halfedge_d:
#ctypedef halfedge_t halfedge_d
#ctypedef face_t face_d
#ctypedef vertex_t vertex_d

LOOP_ALPHA_FACTOR = (np.log(13)-np.log(3))/12


# string key arguments incur a lot of overhead - use index / enums
__insertion_keys = ['vertex', 'halfedge']

DEF INSERTION_KEY_VERTEX = 0
DEF INSERTION_KEY_HALFEDGE = 1

cdef class TrianglesBase(object):
    """
    Base class to let VisGUI layers detect objects which are a bunch of triangles
    
    a derived class should provide suitable attributes/properties for vertices, faces, face_normals,
    and vertex_normals.
    
    See Also: Tesselation class in PYME.recipes.pointcloud
    
    TODO - move me out of here?
    TODO - provide default implementations for face_normals and vertex_normals?
    
    """
    vertices=None
    faces=None
    face_normals=None
    vertex_normals = None
    
    def keys(self):
        """
        returns the names of possible vertex attributes
        """
        raise NotImplementedError('Over-ride in derived class')
    
    def __getitem__(self, item):
        """
        returns a given vertex attribute
        """
        raise NotImplementedError('Over-ride in derived class')

cdef class TriangleMesh(TrianglesBase):

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
        """
        if mesh is not None:
            
            self._vertices = np.copy(mesh._vertices)
            # Flatten to address cython view problem
            _flat_vertices = self._vertices.view(VERTEX_DTYPE2)
            # Set up c views to arrays
            self._set_cvertices(_flat_vertices)
            self._vertex_vacancies = copy.copy(mesh._vertex_vacancies)

            self._faces = np.copy(mesh._faces)
            _flat_faces = self._faces.view(FACE_DTYPE2)
            # Set up c views to arrays
            self._set_cfaces(_flat_faces)
            self._face_vacancies = copy.copy(mesh._face_vacancies)

            self._halfedges = np.copy(mesh._halfedges)
            self._set_chalfedges(self._halfedges)
            self._halfedge_vacancies = copy.copy(mesh._halfedge_vacancies)
        else:
            self.build_from_verts_faces(vertices, faces)

        # Representation of faces by triplets of vertices
        self._faces_by_vertex = None
        self._components_valid = 0
        self._H = None
        self._K = None

        # loop subdivision vars
        self._loop_subdivision_flip_edges = []
        self._loop_subdivision_new_vertices = []

        # Singular edges
        # self._singular_edges = None
        self._singular_edges_valid = 0
        # self._singular_vertices = None
        self._singular_vertices_valid = 0

        # Populate the normals
        #print('populating face normals')
        self._face_normals_valid = 0
        self.face_normals
        #print('populating vertex normals')
        self._vertex_normals_valid = 0
        self.vertex_normals
        #print('done populating normals')

        # Properties we can visualize
        self.vertex_properties = ['x', 'y', 'z', 'component', 'boundary', 'singular', 'curvature_mean', 'curvature_gaussian']
        self.vertex_vector_properties = ['vertex_normals']
        
        self.extra_vertex_data = {}

        self.fix_boundary = 1  # Hold boundary edges in place

        # Is the mesh manifold?
        self._manifold = 0

        # Curvatures
        self._H = None
        self._K = None
        self.smooth_curvature = 0

        self._components_valid = 0

        self.mdh = None

        # Set fix_boundary, etc.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, k):
        # this defers evaluation of the properties until we actually access them, 
        # as opposed to the mappings which stored the values on class creation.
        try:
            res = getattr(self, k)
        except AttributeError:
            try:
                res = self.extra_vertex_data[k]
            except KeyError:
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
    def from_np_stl(cls, triangles_stl, origin=[0,0,0], **kwargs):
        """
        Read from an already-loaded STL stream.


        Parameters
        ----------

        triangles_stl : np.ndarray
            the triangles as a numpy array
        origin: 3-tuple (or array)
            coordinates of the mesh origin (nm). Used to ensure that meshes generated from image isosurfaces line up with those generated from different ROIs / points.
            
        """
        vertices_raw = np.vstack((triangles_stl['vertex0'], 
                                  triangles_stl['vertex1'], 
                                  triangles_stl['vertex2']))

        vertices_raw = vertices_raw + np.array(origin)[None,:]
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
    def curvature_mean(self):
        if self._H is None:
            self.calculate_curvatures()
        return self._H
    
    @property
    def curvature_gaussian(self):
        if self._K is None:
            self.calculate_curvatures()
        return self._K

    @property
    def component(self):
        # if np.all(self._vertices['component'] == -1):
        if not self._components_valid:
            self.find_connected_components()
            self._components_valid = 1
        return self._vertices['component']

    @property
    def vertices(self):
        return self._vertices['position']

    @property
    def vertex_mask(self):
        """
        Returns a mask indicating which vertices are valid
        """

        return self._vertices['halfedge'] != -1

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
        #FIXME - is this simply a duplicate of the .singular property?
        self._update_vertex_locally_manifold()
        return (self._vertices['locally_manifold'] == 0)

    @property
    def singular(self):
        self._update_singular_edge_locally_manifold()
        self._update_singular_vertex_locally_manifold()
        return (self._vertices['locally_manifold'] == 0)

    @property
    def singular_edges(self):
        """
        Get a list of singular edges. A singular edge is any edge shared by
        more than two faces.
        """
        self._update_singular_edge_locally_manifold()
        return np.flatnonzero(self._halfedges['locally_manifold'] == 0)

    @property
    def singular_vertices(self):
        """
        Get a list of singular vertices. This includes endpoints of singular 
        edges, and isolated singular vertices. Isolated singular vertices
        have more than one 1-neighbor ring.
        """
        self._update_singular_edge_locally_manifold()
        self._update_singular_vertex_locally_manifold()
        return np.flatnonzero(self._vertices['locally_manifold'] == 0)

    def _update_all_face_normals(self, recompute=True):
        """
        Update face normals

        If called with recompute=False, only recomputes if face normals have been marked as invalid 
        """
        if recompute or (not self._face_normals_valid):
            triangle_mesh_utils.c_update_all_face_normals(self._faces.shape[0], self._halfedges, self._vertices, self._faces)
        self._face_normals_valid = 1
    
    @property
    def face_normals(self):
        """
        Return the normal of each triangular face.
        """
        self._update_all_face_normals(recompute=False)
        return self._faces['normal'][self._faces['halfedge'] != -1]

    @property
    def face_areas(self):
        """
        Return the area of each triangular face.
        """
        self._update_all_face_normals(recompute=False)
        return self._faces['area']
    
    def _update_all_vertex_neighbours(self, recompute=True):
        """
        Update the vertex neighbour information (neighbours, normals, edge lengths)

        If called with recompute=False, only recomputes if vertex normals have been marked as invalid
        """
        if recompute or (not self._vertex_normals_valid):
            triangle_mesh_utils.c_update_all_vertex_neighbors(self._vertices.shape[0], self._halfedges, self._vertices, self._faces)
            self._vertex_normals_valid = 1

    @property
    def vertex_neighbors(self):
        """
        Return the up to NEIGHBORSIZE neighbors of each vertex.
        """
        
        self._update_all_vertex_neighbours(recompute=False)
        return self._vertices['neighbors']

    @property
    def vertex_normals(self):
        """
        Return the normal of each vertex.
        """
        self._update_all_vertex_neighbours(recompute=False)
        return self._vertices['normal']

    @property
    def valences(self):
        """
        Return the valence of each vertex.
        """
        self._update_all_vertex_neighbours(recompute=False)
        return self._vertices['valence']

    @property
    def manifold(self):
        """
        Checks if the mesh is manifold: Is every edge is shared by exactly two 
        triangles?
        """

        if not self._manifold:
            vertex_mask = (self._halfedges['vertex'] != -1)
            twin_mask = (self._halfedges['twin'] == -1)
            self._manifold = (np.sum(self.singular) == 0) and (np.sum(vertex_mask & twin_mask) == 0)

        return self._manifold

    @property
    def euler_characteristic(self):
        # TODO: Cache on a parameter
        chi = self._calculate_euler_characteristic()
        
        return chi

    @property
    def genus(self):
        return (2-self.euler_characteristic)/2

    @property
    def bbox(self):
        """
        Mesh bounding box. xl, yl, zl, xu, yu, zu
        """
        cdef float xl, xu, yl, yu, zl, zu
        v = self._vertices['position'][self._vertices['halfedge'] != -1]
        xl, xu = np.min(v[:,0]), np.max(v[:,0])
        yl, yu = np.min(v[:,1]), np.max(v[:,1])
        zl, zu = np.min(v[:,2]), np.max(v[:,2])

        return xl, yl, zl, xu, yu, zu

    def keys(self):
        return list(self.vertex_properties) + list(self.extra_vertex_data.keys())

    def _update_vertex_locally_manifold(self):
        # identify boundary vertices
        self._vertices['locally_manifold'] = 1
        boundary_edges = (self._halfedges['twin'] == -1) & (self._halfedges['vertex'] != -1)
        self._vertices['locally_manifold'][self._halfedges['vertex'][boundary_edges]] = 0
        self._vertices['locally_manifold'][self._halfedges['vertex'][self._halfedges['prev'][boundary_edges]]] = 0

        self._singular_vertices_valid = 0

    def _update_singular_vertex_locally_manifold(self, clear=True):
        if not self._singular_vertices_valid:
            
            # identify singular vertices
            if clear:
                # optional so this can be chained with _update_vertex_locally_manifold
                self._vertices['locally_manifold'] = 1

            # vertices connected to singular edges
            non_manifold = self._halfedges['locally_manifold'] == 0
            self._vertices['locally_manifold'][self._halfedges['vertex'][non_manifold]] = 0
            self._vertices['locally_manifold'][self._halfedges['vertex'][self._halfedges['prev'][non_manifold]]] = 0

            # isolated signular vertices
            valence = np.zeros(self._vertices.shape[0],dtype=int)
            valence[self._vertices['halfedge'] == -1] = -1
            self._compute_raw_vertex_valences(valence)
            # print("valence check")
            # print(valence[valence != self._vertices['valence']])
            # print(self._vertices['valence'][valence != self._vertices['valence']])
            self._vertices['locally_manifold'][valence != self._vertices['valence']] = 0

            self._singular_vertices_valid = 1

    # def __update_singular_edge_locally_manifold(self, clear=True):
    #     if not self._singular_edges_valid:
    #         if clear:
    #             # optional so this can be chained with _update_vertex_locally_manifold
    #             self._halfedges['locally_manifold'] = 1

    #         # grab all edges
    #         edges = np.vstack([self._halfedges['vertex'][self._halfedges['prev']],
    #                            self._halfedges['vertex']])

    #         # Sort lo->hi
    #         sorted_edges = np.sort(edges, axis=0)

    #         # find the number of unique elements
    #         _, idxs, counts = np.unique(sorted_edges,axis=1,return_counts=True,return_inverse=True)
    #         singular = np.sum(idxs[None,:] == np.flatnonzero(counts > 2)[:,None],axis=0).astype(bool)

    #         self._halfedges['locally_manifold'][singular & (self._halfedges['vertex'] != -1)] = 0

    #         self._singular_edges_valid = 1

    def _update_singular_edge_locally_manifold(self, clear=True):
        cdef int i
        cdef int v1, v2, _v1, _v2
        cdef int l_v, n_e

        cdef int n_halfedges = self._halfedges.shape[0]
        cdef int n_vertices = self._vertices.shape[0]
        cdef int n_slots = n_vertices*n_vertices

        #cdef int * d

        if not self._singular_edges_valid:
            d = {}

            if clear:
                # optional so this can be chained with _update_vertex_locally_manifold
                self._halfedges['locally_manifold'] = 1
            
            #print('iterating edges')
            # Use a dictionary to keep track of which edges are already assigned twins

            for i in range(n_halfedges):
                if self._chalfedges[i].prev == -1:
                    continue
                _v1 = self._chalfedges[self._chalfedges[i].prev].vertex
                _v2 = self._chalfedges[i].vertex

                if (_v1 == -1) or (_v2 == -1):
                    pass
                else:
                    v1 = min(_v1, _v2)
                    v2 = max(_v1, _v2)

                    slot_idx = v1*n_vertices + v2

                    n_e = d.get(slot_idx, 0)

                    d[slot_idx] = n_e + 1

            for i in range(n_halfedges):
                if self._chalfedges[i].prev == -1:
                    continue
                _v1 = self._chalfedges[self._chalfedges[i].prev].vertex
                _v2 = self._chalfedges[i].vertex

                if (_v1 == -1) or (_v2 == -1):
                    self._chalfedges[i].locally_manifold = -1  # leave unassigned so it does not get included in _update_singular_vertex_locally_manifold()
                else:
                    v1 = min(_v1, _v2)
                    v2 = max(_v1, _v2)

                    slot_idx = v1*n_vertices + v2

                    if d[slot_idx] <=2:    
                        self._chalfedges[i].locally_manifold = 1
                    else:
                        self._chalfedges[i].locally_manifold = 0

            #PyMem_Free(d)

            self._singular_edges_valid = 1

    cdef _compute_raw_vertex_valences(self, np.ndarray valence):
        cdef int n_halfedges = self._halfedges.shape[0]
        cdef int i

        # count the valence
        for i in range(n_halfedges):
            if self._chalfedges[i].vertex == -1:
                continue
            valence[self._chalfedges[i].vertex] += 1

    def _calculate_euler_characteristic(self, component=None):
        if component is not None:
            if not self._components_valid:
                self.find_connected_components()
                self._components_valid = 1
            
            faces = self._faces['halfedge'][self._halfedges['component'][self._faces['halfedge']] == component]
        else:
            faces = self._faces['halfedge'][self._faces['halfedge'] != -1]

        v0 = self._halfedges['vertex'][self._halfedges['prev'][faces]]
        v1 = self._halfedges['vertex'][faces]
        v2 = self._halfedges['vertex'][self._halfedges['next'][faces]]
        faces_by_vertex = np.hstack([v0, v1, v2])

        F = len(faces)
        V = len(set(faces_by_vertex.ravel()))

        # grab all edges
        edges = np.vstack([faces_by_vertex, np.hstack([v1,v2,v0])]).T

        # Sort lo->hi
        sorted_edges = np.sort(edges, axis=1)

        # find the number of unique elements
        E = len(np.unique(sorted_edges,axis=0))
        
        chi = V - E + F
        
        return chi
        
    cdef _set_chalfedges(self, np.ndarray halfedges):
        self._chalfedges = <halfedge_t *> np.PyArray_DATA(halfedges)
        self._n_edges = np.PyArray_DIM(halfedges, 0)
        
    cdef _set_cfaces(self, np.ndarray faces):
        self._cfaces = <face_d *> np.PyArray_DATA(faces)
        self._n_faces = np.PyArray_DIM(faces, 0)
        
    cdef _set_cvertices(self, np.ndarray vertices):
        self._cvertices = <vertex_d *> np.PyArray_DATA(vertices)
        self._n_vertices = np.PyArray_DIM(vertices, 0)
    
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
            
            self._halfedges = np.zeros(n_edges, dtype=HALFEDGE_DTYPE)# .view(np.recarray)
            self._set_chalfedges(self._halfedges)

            self._halfedges[:] = -1  # initialize everything to -1 to start with

            self._halfedges['vertex'] = edges[:, 1]
            
            self._faces = np.zeros(n_faces, dtype=FACE_DTYPE)
            # Flatten to address cython view problem
            _flat_faces = self._faces.view(FACE_DTYPE2)
            # Set up c views to arrays
            self._set_cfaces(_flat_faces)
            self._faces[:] = -1  # initialize everything to -1 to start with

            # Sort the edges lo->hi so we can arrange them uniquely
            # Convert to list of tuples for use with dictionary
            edges_packed = [tuple(e) for e in np.sort(edges, axis=1)]


            #print('iterating edges')
            # Use a dictionary to keep track of which edges are already assigned twins
            d = {}
            for i, e in enumerate(edges_packed):
                if e in d:
                    idx = d.pop(e)
                    if self._halfedges['vertex'][idx] == self._halfedges['vertex'][i]:
                        # Don't assign a halfedge to multivalent edges. Note this doesn't 
                        # account for all multivalent edges, but does prevent double-
                        # assignment of twins.
                        continue
                        
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

    def build_from_verts_faces(self, vertices, faces, clear=False):
        """
        Construct mesh from vertices and faces.
        """
        self._vertices = np.zeros(vertices.shape[0], dtype=VERTEX_DTYPE)
        # Flatten to address cython view problem
        _flat_vertices = self._vertices.view(VERTEX_DTYPE2)
        # Set up c views to arrays
        #print self._flat_vertices.shape
        self._set_cvertices(_flat_vertices)
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

        if clear:
            self._invalidate_cached_properties()
            self._update_all_face_normals()
            self._update_all_vertex_neighbours()

    def _invalidate_cached_properties(self):
        """
        Invalidate properties which have been cached and will now need to be recomputed due to changes in mesh
        """
        self._faces_by_vertex = None
        self._manifold = 0
        # self._singular_edges = None
        # self._singular_vertices = None
        self._singular_edges_valid = 0
        self._singular_vertices_valid = 0
        self._components_valid = 0
        self._face_normals_valid = 0
        self._vertex_normals_valid = 0
        self._H = None
        self._K = None
        #self._E = None
        #self._pE = None

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)
    def _update_face_normals(self, f_idxs):
        """
        Recompute len(f_idxs) face normals.

        Parameters
        ----------
        f_idxs : list or np.array
            List of face indices to recompute.
        """

        if isinstance(f_idxs, list):
            f_idxs = np.int32(f_idxs)
        #triangle_mesh_utils.c_update_face_normals(f_idxs, self._halfedges, self._vertices, self._faces)
        cdef np.int32_t [:] idxs = f_idxs
        
        _update_face_normals(&(idxs[0]), self._chalfedges, <vertex_t *> self._cvertices, <face_t *> self._cfaces, idxs.shape[0])

    def _update_vertex_neighbors(self, v_idxs):
        """
        Recalculate len(v_idxs) vertex neighbors/normals.

        Parameters
        ----------
            v_idxs : list or np.array
                List of vertex indicies indicating which vertices to update.
        """

        if isinstance(v_idxs, list):
            v_idxs = np.int32(v_idxs)
        triangle_mesh_utils.c_update_vertex_neighbors(v_idxs, self._halfedges, self._vertices, self._faces)

    def _resize(self, vec, axis=0, return_orig=False, skip_entries=False, key=None, new_items=1):
        """
        Increase the size of an input vector 1.5x, keeping all active data.

        Parameters
        ---------
            vec : np.array
                Vector to expand, -1 stored in unused entries.
            axis : int
                Axis (0 or 1) along which to resize vector (only works for 2D 
                arrays).
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
        new_size[axis] = int(max(1.5*new_size[axis], new_size[axis]+ new_items + 1) + 0.5)
        new_size = tuple(new_size)
        
        # Allocate memory for new array
        new_vec = np.empty(new_size, dtype=dt)

        # Assign values to the new array
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

    def calculate_curvatures(self):
        # TODO - compare / consolidate with ch-shrinkwrap version
        dN = 0.1
        self._H = np.zeros(self._vertices.shape[0])
        self._K = np.zeros(self._vertices.shape[0])
        I = np.eye(3)
        areas = np.zeros(self._vertices.shape[0])
        for iv in range(self._vertices.shape[0]):
            if self._vertices['halfedge'][iv] == -1:
                continue

            # Vertex and its normal
            vi = self._vertices['position'][iv,:]
            Nvi = self._vertices['normal'][iv,:]

            p = I - Nvi[:,None]*Nvi[None,:] # np.outer(Nvi, Nvi)
                
            # vertex nearest neighbors
            neighbors = self._vertices['neighbors'][iv]
            neighbor_mask = (neighbors != -1)
            neighbor_vertices = self._halfedges['vertex'][neighbors]
            vjs = self._vertices['position'][neighbor_vertices[neighbor_mask]]

            # Neighbor vectors & displaced neighbor tangents
            dvs = vjs - vi[None,:]

            # radial weighting
            r_sum = np.sum(1./np.sqrt((dvs*dvs).sum(1)))

            # Norms
            dvs_norm = np.sqrt((dvs*dvs).sum(1))

            # Hats
            dvs_hat = dvs/dvs_norm[:,None]

            # Tangents
            T_thetas = np.dot(p,-dvs.T).T
            Tijs = T_thetas/np.sqrt((T_thetas*T_thetas).sum(1)[:,None])
            Tijs[np.sum(T_thetas,axis=1) == 0, :] = 0

            # Edge normals subtracted from vertex normals
            Ni_diffs = np.sqrt(2. - 2.*np.sqrt(1.-((Nvi[None,:]*dvs_hat).sum(1))**2))

            k = 2.*np.sign((Nvi[None,:]*(-dvs)).sum(1))*Ni_diffs/dvs_norm
            w = (1./dvs_norm)/r_sum

            Mvi = (w[None,:,None]*k[None,:,None]*Tijs.T[:,:,None]*Tijs[None,:,:]).sum(axis=1)

            # Solve the eigenproblem in closed form
            m00 = Mvi[0,0]
            m01 = Mvi[0,1]
            m02 = Mvi[0,2]
            m11 = Mvi[1,1]
            m12 = Mvi[1,2]
            m22 = Mvi[2,2]

            # Here we use the fact that Mvi is symnmetric and we know
            # one of the eigenvalues must be 0
            p = -m00*m11 - m00*m22 + m01*m01 + m02*m02 - m11*m22 + m12*m12
            q = m00 + m11 + m22
            r = np.sqrt(4*p + q*q)
            
            # Eigenvalues
            l1 = 0.5*(q-r)
            l2 = 0.5*(q+r)

            k_1 = 3.*l1 - l2 #e[0] - e[1]
            k_2 = 3.*l2 - l1 #e[1] - e[0]
            self._H[iv] = 0.5*(k_1 + k_2)
            self._K[iv] = k_1*k_2

        if self.smooth_curvature:    
            self._H = self.smooth_per_vertex_data(self._H)
            self._K = self.smooth_per_vertex_data(self._K)
        
    def smooth_per_vertex_data(self, data):
        # replace a vertex value with the average of that value and it's neighbours. TODO - add some form of weighting
        out = np.zeros_like(data)
        for iv in range(self._vertices.shape[0]):
            # Vertex position
            #vi = self._vertices['position'][iv,:]
                
            # vertex nearest neighbors
            neighbors = self._vertices['neighbors'][iv]
            neighbor_vertices = self._halfedges['vertex'][neighbors[neighbors != -1]]
            
            #vjs = self._vertices['position'][neighbor_vertices]
            
            #TODO add weightings ...
            out[iv] = (data[iv] + data[neighbor_vertices].sum())/float(1 + len(neighbor_vertices))
            
        return out
            

    cdef bint _check_neighbour_twins(self, int vertex_id):
        cdef int i
        cdef np.int32_t *neighbours = &(self._cvertices[vertex_id].neighbor0)
        for i in range(NEIGHBORSIZE):
            nn = neighbours[i]
            if (nn != -1) and (self._chalfedges[nn].twin == -1):
                return 0
            
        return 1

    cdef int _face_delete(self, np.int32_t _edge):
        """
        Delete a face defined by the halfedge _edge.

        Note that this does not account for adjusting halfedges, twins.

        Parameters
        ----------
            edge : int
                One self._halfedge index of the edge defining a face to delete.
        """
        cdef halfedge_t *curr_edge
        
        if _edge == -1:
            return 0
        
        curr_edge = &self._chalfedges[_edge]
        if curr_edge.vertex == -1:
            return 0
        
        self._cfaces[curr_edge.face].halfedge = -1
        self._cfaces[curr_edge.face].normal0 = -1
        self._cfaces[curr_edge.face].normal1 = -1
        self._cfaces[curr_edge.face].normal2 = -1
        self._cfaces[curr_edge.face].area = -1
        self._cfaces[curr_edge.face].component = -1

        self._face_vacancies.append(curr_edge.face)
        
        self._edge_delete(curr_edge.next)
        self._edge_delete(curr_edge.prev)
        self._edge_delete(_edge)

        self._invalidate_cached_properties()

        return 1

    cdef int _edge_delete(self, np.int32_t _edge):
        """
        Delete edge _edge in place from self._halfedges.

        Note that this does not account for adjusting halfedges, twins.

        Parameters
        ----------
            edge : int
                One self._halfedge index of the edge to delete.
        """
        if (_edge == -1):
            return 0
        
        self._chalfedges[_edge].vertex = -1
        self._chalfedges[_edge].face = -1
        self._chalfedges[_edge].twin = -1
        self._chalfedges[_edge].next = -1
        self._chalfedges[_edge].prev = -1
        self._chalfedges[_edge].length = -1
        self._chalfedges[_edge].component = -1
        self._chalfedges[_edge].locally_manifold = -1

        self._halfedge_vacancies.append(_edge)

        self._invalidate_cached_properties()

        return 1

    cdef int _vertex_delete(self, np.int32_t v_idx):
        if v_idx == -1:
            return 0

        self._cvertices[v_idx].position0 = -1
        self._cvertices[v_idx].position1 = -1
        self._cvertices[v_idx].position2 = -1
        self._cvertices[v_idx].normal0 = -1
        self._cvertices[v_idx].normal1 = -1
        self._cvertices[v_idx].normal2 = -1
        self._cvertices[v_idx].halfedge = -1
        self._cvertices[v_idx].valence = -1
        self._cvertices[v_idx].neighbor0 = -1
        self._cvertices[v_idx].neighbor1 = -1
        self._cvertices[v_idx].neighbor2 = -1
        self._cvertices[v_idx].neighbor3 = -1
        self._cvertices[v_idx].neighbor4 = -1
        self._cvertices[v_idx].neighbor5 = -1
        self._cvertices[v_idx].neighbor6 = -1
        self._cvertices[v_idx].neighbor7 = -1
        self._cvertices[v_idx].neighbor8 = -1
        self._cvertices[v_idx].neighbor9 = -1
        self._cvertices[v_idx].neighbor10 = -1
        self._cvertices[v_idx].neighbor11 = -1
        self._cvertices[v_idx].neighbor12 = -1
        self._cvertices[v_idx].neighbor13 = -1
        self._cvertices[v_idx].neighbor14 = -1
        self._cvertices[v_idx].neighbor15 = -1
        self._cvertices[v_idx].neighbor16 = -1
        self._cvertices[v_idx].neighbor17 = -1
        self._cvertices[v_idx].neighbor18 = -1
        self._cvertices[v_idx].neighbor19 = -1
        self._cvertices[v_idx].component = -1
        self._cvertices[v_idx].locally_manifold = -1

        self._vertex_vacancies.append(v_idx)

        self._invalidate_cached_properties()

        return 1

    def _get_insertion_slots(self, int n_slots, el_arr, el_vacancies, key_idx, compact=False):
        """ get an empty slot to insert into, resizing if needed"""
        
        if len(el_vacancies) <= n_slots:
            # not enough vacant slots, resize
            key = __insertion_keys[key_idx]
            el_arr = self._resize(el_arr, skip_entries=compact, key=key, new_items=n_slots)
            #TODO - invalidate neighbours, vertex_halfedges etc ???
            
            # NOTE: If we search by a different key next time, el_vacancies will
            # still contain the vacant positions when searching on the previous key.
            el_vacancies = self._update_vacancies(el_arr, key)
            
        idx = el_vacancies[-n_slots:]
        el_vacancies = el_vacancies[:-n_slots]

        if idx == -1:
            raise ValueError('Index cannot be -1.')
        
        return idx, el_arr, el_vacancies

    def _update_vacancies(self, el_arr, key):
        if len(el_arr[key].shape) > 1:
            # NOTE: If we search on self._vertices['position'] and one position is [-1,-1,-1],
            # which is not unreasonable, this will replace that vertex. Hence we provide the
            # option to search and insert on different keys.
            el_vacancies = [int(x) for x in np.argwhere(np.all(el_arr[key] == -1, axis=1))]
        else:
            # el_vacancies = [int(x) for x in np.flatnonzero(el_arr[key] == -1)]
            el_vacancies = np.flatnonzero(el_arr[key] == -1).tolist()

        return el_vacancies

    def _update_all_vacancies(self):
        self._halfedge_vacancies = self._update_vacancies(self._halfedges, __insertion_keys[INSERTION_KEY_VERTEX])
        self._vertex_vacancies = self._update_vacancies(self._vertices, __insertion_keys[INSERTION_KEY_HALFEDGE])
        self._face_vacancies = self._update_vacancies(self._faces, __insertion_keys[INSERTION_KEY_HALFEDGE])

    

    cdef _populate_edge(self, int idx, int vertex, int prev=-1, int next=-1, int face=-1, int twin=-1, int locally_manifold=1):
        # TODO - make a macro??
        self._chalfedges[idx].vertex = vertex
        self._chalfedges[idx].prev = prev
        self._chalfedges[idx].next = next
        self._chalfedges[idx].face = face
        self._chalfedges[idx].twin = twin
        self._chalfedges[idx].locally_manifold = locally_manifold

    def new_edges(self, int n_edges):
        idx, self._halfedges, self._halfedge_vacancies = self._get_insertion_slots(n_edges, self._halfedges, self._halfedge_vacancies, key_idx=INSERTION_KEY_VERTEX)
        self._set_chalfedges(self._halfedges)

        return np.array(idx, np.int32)

    def new_vertices(self, int n_vertices):
        idx, self._vertices, self._vertex_vacancies = self._get_insertion_slots(n_vertices, self._vertices, self._vertex_vacancies, key_idx=INSERTION_KEY_HALFEDGE)
        self._set_cvertices(self._vertices)

        self._invalidate_cached_properties()

        return np.array(idx, np.int32)

    def new_faces(self, int n_faces):
        idx, self._faces, self._face_vacancies = self._get_insertion_slots(n_faces, self._faces, self._face_vacancies, key_idx=INSERTION_KEY_HALFEDGE)
        self._set_cfaces(self._faces)

        return np.array(idx, np.int32)
    
    cpdef int edge_flip(self, np.int32_t _curr, bint live_update=1):
        cdef int ret
        ret = remesh_edge_flip(self._chalfedges, <vertex_t *>&self._cvertices[0], <face_t *> &self._cfaces[0], _curr, self._n_edges, live_update)
        self._invalidate_cached_properties()
        return ret
        
    def split_edges(self, float split_threshold):
        cdef int split_count = 0
        cdef int i
        cdef int e
        cdef int n_halfedges = self._halfedges.shape[0]
        cdef float *n1
        cdef float *n2
        cdef float nd
        cdef int n_edge_idx, n_face_idx, n_vertex_idx
        #self.face_normals
        #self.vertex_normals

        cdef int* edges_to_split = <int*>PyMem_Malloc(n_halfedges*sizeof(int))
        if not edges_to_split:
            raise MemoryError()
        cdef int* twin_split = <int*>PyMem_Malloc(n_halfedges*sizeof(int))
        if not twin_split:
            raise MemoryError()

        for i in range(n_halfedges):
            twin_split[i] = 0
        
        for i in range(n_halfedges):
            if (not twin_split[i]) and (self._chalfedges[i].vertex != -1): # and (self._chalfedges[i].length > split_threshold):
                # n1 = &self._cvertices[self._chalfedges[i].vertex].normal0
                # n2 = &self._cvertices[self._chalfedges[self._chalfedges[i].prev].vertex].normal0
                # nd = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2]
                nd = 1.0
                if (self._chalfedges[i].length > (nd*split_threshold)):
                    if self._chalfedges[i].twin != -1:
                        twin_split[self._chalfedges[i].twin] = 1
                    edges_to_split[split_count] = i
                    split_count += 1

        n_edges = self.new_edges(int(split_count*6))
        n_edge_idx = 0
        n_faces = self.new_faces(int(split_count*2))
        n_face_idx = 0
        n_vertices = self.new_vertices(int(split_count))
        n_vertex_idx = 0
        #print(self._halfedges[n_edges])

        for i in range(split_count):
            e = edges_to_split[i]
            #print(i, e, n_edge_idx, n_edges)
            # self.edge_split_2(e, 
            #                  <np.int32_t *> np.PyArray_DATA(n_edges), 
            #                  <np.int32_t *> np.PyArray_DATA(n_vertices), 
            #                  <np.int32_t *> np.PyArray_DATA(n_faces), 
            #                  n_edge_idx, n_vertex_idx, n_face_idx)

            remesh_split_edge(self._chalfedges, n_halfedges, <vertex_t *> self._cvertices, <face_t *>self._cfaces, int(e),
                              <np.int32_t *> np.PyArray_DATA(n_edges), 
                             <np.int32_t *> np.PyArray_DATA(n_vertices), 
                             <np.int32_t *> np.PyArray_DATA(n_faces), 
                             n_edge_idx, n_vertex_idx, n_face_idx, 1, 0)
            #self.edge_split(e)
            n_edge_idx += 6
            n_face_idx += 2
            n_vertex_idx += 1

        PyMem_Free(edges_to_split)
        PyMem_Free(twin_split)
                
        #print('Split count: %d' % (split_count))
        return split_count
    
    def collapse_edges(self, float collapse_threshold):
        cdef int collapse_count = 0
        cdef int collapse_fails = 0
        cdef int i
        cdef int n_halfedges = self._halfedges.shape[0]
        cdef float *n1
        cdef float *n2
        cdef float nd
        
        #find which vertices are locally manifold
        self._update_vertex_locally_manifold()
        
        for i in range(n_halfedges):
            if (self._chalfedges[i].vertex != -1) and (self._chalfedges[i].length < collapse_threshold):
                n1 = &self._cvertices[self._chalfedges[i].vertex].normal0
                n2 = &self._cvertices[self._chalfedges[self._chalfedges[i].prev].vertex].normal0
                # dot product of the normals
                nd = n1[0]*n2[0] + n1[1]*n2[1] + n1[2]*n2[2]
                if (self._chalfedges[i].length < nd*nd*collapse_threshold):
                    #collapse_ret = self.edge_collapse(i)
                    collapse_ret = remesh_edge_collapse(self._chalfedges, self._n_edges, <vertex_t *> self._cvertices, <face_t*> self._cfaces, i, 1)
                    collapse_count += collapse_ret
                    collapse_fails += (1-collapse_ret)

        self._update_all_vacancies() #we might have deleted edges, vertices, etc ....
        self._invalidate_cached_properties()
        #print('Collapse count: ' + str(collapse_count) + '[' + str(collapse_fails) +' failed]')
        
        return collapse_count

    cpdef int regularize(self):
        """
        Adjust vertices so they tend toward valence VALENCE 
        (or BOUNDARY_VALENCE for boundaries).
        """

        cdef int i, flip_count, target_valence, r, failed_flip_count
        cdef halfedge_t *curr_edge
        cdef halfedge_t *twin_edge
        cdef np.int32_t _twin
        cdef int v1, v2, v3, v4, score_post, score_pre

        flip_count = 0
        failed_flip_count = 0

        # Do a single pass over all active edges and flip them if the flip minimizes the deviation of vertex valences
        # for i in np.arange(len(self._halfedges['vertex'])):
        #    if (self._halfedges['vertex'][i] < 0):
        #        continue
        for i in range(self._halfedges.shape[0]):# np.flatnonzero(self._halfedges['vertex'] != -1).astype(np.int32):
            curr_edge = &self._chalfedges[i]
            
            if curr_edge.vertex == -1:
                continue
            
            _twin = curr_edge.twin

            v2, v4 = 0, 0
            target_valence = VALENCE
            if _twin == -1:
                # boundary
                target_valence = BOUNDARY_VALENCE
            else:
                twin_edge = &self._chalfedges[_twin]
                v2 = self._cvertices[twin_edge.vertex].valence - target_valence  # pre-flip 
                v4 = self._cvertices[self._chalfedges[twin_edge.next].vertex].valence - target_valence  # post-flip

            v1 = self._cvertices[curr_edge.vertex].valence - target_valence  # pre-flip
            v3 = self._cvertices[self._chalfedges[curr_edge.next].vertex].valence - target_valence  # post-flip

            # Check valence deviation from VALENCE (or
            # BOUNDARY_VALENCE for boundaries) pre- and post-flip
            #score_pre = abs([v1,v2,v3,v4]).sum()
            #score_post = np.abs([v1-1,v2-1,v3+1,v4+1]).sum()
            
            score_pre = abs(v1) + abs(v2) + abs(v3) + abs(v4)
            score_post = abs(v1-1) + abs(v2-1) + abs(v3+1) + abs(v4+1)

            if score_post < score_pre:
                # Flip minimizes deviation of vertex valences from VALENCE (or
                # BOUNDARY_VALENCE for boundaries)
                r = self.edge_flip(i)
                flip_count += r
                failed_flip_count += (1-r)
                
        #print('Flip count: %d [%d failed]' % (flip_count, failed_flip_count))
        return flip_count

    cpdef int relax(self, float l=1, int n=1):
        return remesh_relax(self._chalfedges, <vertex_t *> self._cvertices, self._n_vertices, <face_t *> self._cfaces, self._n_faces, l, n)


    def remesh(self, int n=5, float target_edge_length=-1, float l=0.5, int n_relax=10):
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
        cdef int k, i, split_count, collapse_count, collapse_ret, collapse_fails, n_singular
        #cdef float target_edge_length
        
        cdef int n_halfedges = self._halfedges.shape[0]
        cdef int n_vertices = self._vertices.shape[0]

        if (target_edge_length < 0):
            # Guess edge_length
            target_edge_length = np.mean(self._halfedges['length'][self._halfedges['length'] != -1])
            
        cdef float split_threshold = 1.33*target_edge_length
        cdef float collapse_threshold = 0.8*target_edge_length
        
        # modify cutoffs so we don't constantly undo ourselves ...
        # TOD - should we regularize between split and collapse?
        #cdef float split_threshold = 1.41*target_edge_length
        #cdef float collapse_threshold = 0.7*target_edge_length
        
        #self.regularize()

        self._singular_edges_valid = 0
        n_singular = self.singular_edges.shape[0]
        #f"# singular: {n_singular}")

        for k in range(n):
            # 1. Split all edges longer than (4/3)*target_edge_length at their midpoint.
            # split_count = 0
            # for i in range(n_halfedges):
            #     if (self._chalfedges[i].vertex != -1) and (self._chalfedges[i].length > split_threshold):
            #         self.edge_split(i)
            #         split_count += 1
            # print('Split count: %d' % (split_count))
            
            ct = self.split_edges(split_threshold)
            #while (ct > 0):
            #    # Keep spliting until we can't do any more splits (note - may be more effective to just run everything a couple more times)
            #    ct = self.split_edges(split_threshold)

            self._singular_edges_valid = 0
            n_singular = self.singular_edges.shape[0]
            #print(f"# singular: {n_singular}")
            
            ct = self.regularize()
            while (ct > 0):
                 # Keep flipping until we can't do any more splits (note - may be more effective to just run everything a couple more times)
                 ct = self.regularize()

            self._singular_edges_valid = 0
            n_singular = self.singular_edges.shape[0]
            #print(f"# singular: {n_singular}")
            
            # 2. Collapse all edges shorter than (4/5)*target_edge_length to their midpoint.
            # collapse_count = 0
            # collapse_fails = 0
            #
            # #find which vertices are locally manifold
            # # TODO - move this to a helper function
            # self._vertices['locally_manifold'] = 1
            # self._vertices['locally_manifold'][self._halfedges['vertex'][self._halfedges['twin'] == -1]] = 0
            #
            # for i in range(n_halfedges):
            #     if (self._chalfedges[i].vertex != -1) and (self._chalfedges[i].length < collapse_threshold):
            #         collapse_ret = self.edge_collapse(i)
            #         collapse_count += collapse_ret
            #         collapse_fails += (1-collapse_ret)
            # print('Collapse count: ' + str(collapse_count) + '[' + str(collapse_fails) +' failed]')
            
            ct = self.collapse_edges(collapse_threshold)

            self._singular_edges_valid = 0
            n_singular = self.singular_edges.shape[0]
            #print(f"# singular: {n_singular}")
            # while (ct > 0):
            #     # Keep collapsing until we can't do any more splits (note - may be more effective to just run everything a couple more times)
            #     ct = self.collapse_edges(collapse_threshold)

            # 3. Flip edges in order to minimize deviation from VALENCE.
            
            # find which vertices are locally manifold
            self._update_vertex_locally_manifold()
            
            ct = self.regularize()
            _n_flip = 0
            while (ct > 0) and (_n_flip < 20):
                 # Keep flipping until we can't do any more splits (note - may be more effective to just run everything a couple more times)
                 ct = self.regularize()
                 _n_flip += 1


            self._singular_edges_valid = 0
            n_singular = self.singular_edges.shape[0]
            #print(f"# singular: {n_singular}")

            # 4. Relocate vertices on the surface by tangential smoothing.
            self.relax(l=l, n=n_relax)

            self._singular_edges_valid = 0
            n_singular = self.singular_edges.shape[0]
            #print(f"# singular: {n_singular}")

        # Let's double-check the mesh manifoldness
        self._manifold = 0
        self.manifold

    def upsample(self, int n=1):
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
        cdef int n_edge_idx, n_face_idx, n_vertex_idx, k
        cdef int n_halfedges = self._halfedges.shape[0]

        for k in range(n):
            # Reset
            split_edges = {}
            self._loop_subdivision_new_vertices = []
            self._loop_subdivision_flip_edges = []
            new_vertex_idxs = []

            # Mark the current edges as old
            edges_to_split = np.where(self._halfedges['vertex'] != -1)[0]
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
            split_count = len(edges_to_split)
            n_edges = self.new_edges(int(split_count*6))
            n_edge_idx = 0
            n_faces = self.new_faces(int(split_count*2))
            n_face_idx = 0
            n_vertices = self.new_vertices(int(split_count))
            n_vertex_idx = 0

            for j, i in enumerate(edges_to_split):
                if i in split_edges:
                    continue
                
                split_edges[self._halfedges['twin'][i]] = i
                new_vertex_idxs.append(j)
                #self.edge_split(i, upsample=True)
                # self.edge_split_2(i, 
                #              <np.int32_t *> np.PyArray_DATA(n_edges), 
                #              <np.int32_t *> np.PyArray_DATA(n_vertices), 
                #              <np.int32_t *> np.PyArray_DATA(n_faces), 
                #              n_edge_idx, n_vertex_idx, n_face_idx, 1,1)
                remesh_split_edge(self._chalfedges, n_halfedges, <vertex_t *> self._cvertices, <face_t *>self._cfaces, int(i),
                              <np.int32_t *> np.PyArray_DATA(n_edges), 
                             <np.int32_t *> np.PyArray_DATA(n_vertices), 
                             <np.int32_t *> np.PyArray_DATA(n_faces), 
                             n_edge_idx, n_vertex_idx, n_face_idx, 1, 0)
    
                n_edge_idx += 6
                n_face_idx += 2
                n_vertex_idx += 1
            
            # 2. Flip any new edge that touches an old vertex and a new vertex
            for e in self._loop_subdivision_flip_edges:
                if self._halfedges['vertex'][e] in self._loop_subdivision_new_vertices:
                    continue
                self.edge_flip(e)

            # Get any boundary vertices
            boundary_vertex_idxs = self._halfedges['vertex'][(self._halfedges['twin'] == -1)]
            boundary_vertex_positions = self._vertices['position'][boundary_vertex_idxs]

            # Copy new position values into self._vertices['position']
            self._vertices['position'][old_vertex_idxs] = old_vertex_positions
            self._vertices['position'][self._loop_subdivision_new_vertices] = new_vertex_positions[new_vertex_idxs]

            # Restore boundary vertex positions
            self._vertices['position'][boundary_vertex_idxs] = boundary_vertex_positions

        self._manifold = 0

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
            #self.edge_collapse(_edge)
            remesh_edge_collapse(self._chalfedges, self._n_edges, <vertex_t *> self._cvertices, <face_t*> self._cfaces, _edge, 1)
            
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

        self._update_all_vacancies()
        self._invalidate_cached_properties()

        self._manifold = 0

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
                    if self._vertices['component'][curr_edge['vertex']] != component:
                        search_vertices.append(curr_edge['vertex'])
                    self._halfedges['component'][_edge] = component
                    self._halfedges['component'][curr_edge['prev']] = component
                    self._halfedges['component'][curr_edge['next']] = component
                    self._faces['component'][curr_edge['face']] = component
                    self._vertices['component'][curr_edge['vertex']] = component
            
            # Increment the label as any vertices not included in this
            # iteration's search will be part of another component.
            component += 1

    def area(self, component=-1):
        """
        Return surface area of the mesh (component) in nm^2.
        """
        if component == -1:
            faces = np.flatnonzero(self._faces['halfedge']!=-1)
        else:
            faces = np.flatnonzero(self._faces['component']==component)

        return np.sum(self._faces['area'][faces])

    def volume(self, component=-1):
        """
        Return volume of the mesh (component) in nm^3.

        Sum the signed volumes of tetrahedrons formed by the faces and the origin.

        Cha Zhang and Tsuhan Chen. "Efficient Feature Extraction for 2D/3D Objects in Mesh Representation." 
        In Proceedings 2001 International Conference on Image Processing (Cat. No.01CH37205), 2:935–38. 
        Thessaloniki, Greece: IEEE, 2001. https://doi.org/10.1109/ICIP.2001.958278.
        """
        
        # TODO?? - more component refactoring - e.g. a component object / class / iterator?
        if component == -1:
            faces = np.flatnonzero(self._faces['halfedge']!=-1)
        else:
            faces = np.flatnonzero(self._faces['component']==component)

        faces = self._faces['halfedge'][faces]
        v0 = self._halfedges['vertex'][self._halfedges['prev'][faces]]
        v1 = self._halfedges['vertex'][faces]
        v2 = self._halfedges['vertex'][self._halfedges['next'][faces]]

        p0 = self._vertices['position'][v0]
        p1 = self._vertices['position'][v1]
        p2 = self._vertices['position'][v2]

        t0 = p2[:,0]*p1[:,1]*p0[:,2]
        t1 = -p1[:,0]*p2[:,1]*p0[:,2]
        t2 = -p2[:,0]*p0[:,1]*p1[:,2]
        t3 = p0[:,0]*p2[:,1]*p1[:,2]
        t4 = p1[:,0]*p0[:,1]*p2[:,2]
        t5 = -p0[:,0]*p1[:,1]*p2[:,2]

        volume = (1.0/6.0)*(t0+t1+t2+t3+t4+t5)

        return volume.sum()

    def keep_components_by_volume(self, min_size=0, max_size=1e9):
        # Find the connected components
        self.find_connected_components()

        # Get unique component values
        coms = np.unique(self._faces['component'][self._faces['component']!=-1])

        # Get volumes in nanometers^3
        sizes = np.array([self.volume(c) for c in coms])

        # keep components within size range
        kept_coms = coms[(sizes > min_size) & (sizes < max_size)]

        self.keep_components(kept_coms)


    def keep_largest_connected_component(self, n=1):
        # Find the connected components
        self.find_connected_components()

        # Get n largest connected components
        com, counts = np.unique(self._vertices['component'][self._vertices['component']!=-1], return_counts=True)
        # max_count = np.argmax(counts)
        counts_sorted = np.argsort(counts)
        # max_com = com[max_count]
        max_coms = com[counts_sorted[-n:]]
        
        self.keep_components(max_coms)

    def remove_inner_surfaces(self, max_mean_curvature=0.1):
        """
        Isosurfacing of a closed membrane will often generate two surfaces - one on the
        inside of the structure and one on the outside. The inner and outer surfaces can be
        distinguished on the basis of mean curvature, with outer surfaces having +ve overall
        mean curvature and inside surfaces having -ve overall mean curvature. This function
        culls all surfaces having -ve mean curvature.
        
        To avoid calculating component curvatures twice, this function also attempts to remove
        spurious surfaces (such as those generated by a single voxel > threshold) which can be identified
        by having an excessively large +ve mean curvature.
        
        Parameters
        ----------
        
        max_mean_curvature: float [default=0.1 nm-1]
                    a mean curvature value above which a surface is deemed to be spurious
        
        Returns
        -------

        """
        comps = np.unique(self.component) #find the unique components
        comps = comps[comps >=0] #discard the -1, or 'invalid' component
        
        cc = [(i, np.median(self.curvature_mean[self.component==i])) for i in comps]
        self.keep_components([i for i, c in cc if (c > 0) and (c < max_mean_curvature)])
        
    
    def keep_components(self, components):
        """
        Discard all but the indicated components. Used for pruning the mesh of both
        small objects, and inner surfaces.
        
        Parameters
        ----------
        components : ndarray / list of int
                labels of components to keep

        Returns
        -------

        """
        components = np.array(components)
        
        # Remove the smaller components
        # _vertices = np.where((self._vertices['component'] != max_com))[0]
        _vertices = np.flatnonzero((self._vertices['component'][:,None] != components[None, :]).prod(1))
        # _edges = np.where((self._halfedges['component'] != max_com))[0]
        _edges = np.flatnonzero((self._halfedges['component'][:,None] != components[None, :]).prod(1))
        _edges_with_twins = _edges[self._halfedges['twin'][_edges] != -1]
        # _faces = np.where((self._faces['component'] != max_com))[0]
        _faces = np.flatnonzero((self._faces['component'][:,None] != components[None, :]).prod(1))

        # Maybe don't delete some of these? (If there are singular elements in our mesh.)
        _kept_edges = np.array(list(set(self._halfedges['twin'][_edges_with_twins])-set(_edges)))
        if np.any(_kept_edges):
            _deleted_vertices = np.array(list(set(_vertices)-set(self._halfedges['vertex'][_kept_edges])))
            self._halfedges['twin'][_kept_edges] = -1
        else:
            _deleted_vertices = _vertices
        
        # Delete vertices
        self._vertices[_deleted_vertices] = -1
        self._halfedges[_edges] = -1
        self._faces[_faces] = -1

        # Mark deleted faces as available
        self._vertex_vacancies = list(set(self._vertex_vacancies).union(set(_deleted_vertices)))
        self._halfedge_vacancies = list(set(self._halfedge_vacancies).union(set(_edges)))
        self._face_vacancies = list(set(self._face_vacancies).union(set(_faces)))

        if np.any(_kept_edges):
            # Update newly orphaned edges
            self._update_face_normals(list(set(self._halfedges['face'][_kept_edges])))
            self._update_vertex_neighbors(list(set(self._halfedges['vertex'][_kept_edges])))
        
        self._invalidate_cached_properties()

    cdef _find_boundary_polygons(self, np.ndarray boundary_polygons, np.ndarray boundary_edges):
        """
        Return array of closed boundary polygons.

        We assume boundary_edges is the exhaustive list of halfedges on a boundary.
        """
        cdef int n_edges = boundary_edges.shape[0]
        cdef int max_iters = 6*n_edges  # FIXME: Not clear what this upper bound should be. Seems low.
        cdef int i, j, k, poly, safety, n_poly, new_poly, n_left
        cdef np.int32_t _orig, _curr, _twin
        cdef halfedge_t *curr_edge

        # keep track of already visited edges in components
        # later, keep track of vertex valences in components
        # this is predicated on vertices showing up in only one polygon,
        # which should be the case if we have removed singularities
        self._components_valid = 0
        for i in range(n_edges):
            curr_edge = &self._chalfedges[boundary_edges[i]]
            curr_edge.component = 0
            self._cvertices[curr_edge.vertex].component = 0

        poly = 0  # polygon number

        # loop over all of the boundary edges
        for i in range(n_edges):
            if (self._chalfedges[boundary_edges[i]].component == 1):
                continue
            
            # if this edge is not included in existing polygons, create a new one
            k = 0  # element in polygon
            _curr = boundary_edges[i]
            _orig = _curr

            # traverse around boundary until we return to _orig
            safety = 0  # safety counter
            while True:
                if safety >= max_iters:
                    # Safety release valve so we don't iterate forever
                    print("ERROR: boundary not found")
                    break

                curr_edge = &self._chalfedges[_curr]
                _twin = curr_edge.twin

                if _twin == -1:
                    if k >= n_edges:
                        print("ERROR: Not enough space allocated for this polygon.")
                        break
                    # we're on a boundary, mark it as part of the polygon...
                    boundary_polygons[poly,k] = _curr
                    curr_edge.component = 1
                    # ...also increment the number of incident boundary edges on its vertices
                    self._cvertices[curr_edge.vertex].component += 1
                    self._cvertices[self._chalfedges[curr_edge.prev].vertex].component += 1

                    k += 1

                    # ...then traverse
                    _curr = curr_edge.next
                else:
                    # traverse
                    _curr = self._chalfedges[_twin].next

                if _curr == _orig:
                    # we closed the loop
                    break

                safety += 1
            
            if k == 0:
                print("WARNING: Polygon of length 0????")
                continue
            elif k == 1:
                print("WARNING: Polygons should be of length >= 2. Perhaps there is a singularity in the mesh?")
                poly += 1
            else:
                # Deal with pinched polygons
                # We keep disconnecting pinched polygons from the original polygon and then checking the
                # disconnected polygon for further pinches. The function _disconnect_pinched_polygons is
                # set up s.t. there will be no pinched polygons in the poly entry once run, only in the
                # disconnected polygon in a later entry
                #
                # TODO: Is this necessary? A pinched polygon requires a singular vertex. We should simply
                # call _remove_singularities() before running this function.
                n_poly = poly
                n_left = 1
                while n_left > 0:
                    n_left -= 1
                    new_poly = self._disconnect_pinched_polygons(boundary_polygons, poly, n_poly, k)
                    n_left += new_poly
                    n_poly += new_poly
                    poly += 1  # net increment poly by the number of new polygons we created un-pinching


    cdef int _disconnect_pinched_polygons(self, np.ndarray boundary_polygons, int curr_poly, int n_poly, int n_edges):
        cdef np.int32_t _orig, _curr
        cdef int j, jj, kk, n_new_polygons
        cdef halfedge_t *curr_edge

        _orig = -1
        n_new_polygons = 0

        # check to see if any vertices have more than one boundary edge incident on a vertex
        j = 0
        while boundary_polygons[curr_poly,j] != -1:
            curr_edge = &self._chalfedges[boundary_polygons[curr_poly,j]]
            if self._cvertices[curr_edge.vertex].component > 2:
                # we've hit the case were the boundary pinches in on itself. the twin->next traverse allows this
                # (picture a square boundary with a triangular boundary coming off one of its corners).

                # the good news is everything is wound consistently, so we just have to clip from here to the next
                # instance of this vertex as a new polygon.

                if (_orig != -1) and (curr_edge.vertex == _orig):
                    # we previously marked this vertex, finish splitting this into a new polygon
                    n_new_polygons += 1  # add another polygon
                    
                    print(f"New polygon: {_orig} {self._chalfedges[kk].vertex}")
                    print(f"{boundary_polygons[curr_poly,:14]}")
                    for jj in range(kk,j+1):
                        # copy into the new polygon
                        boundary_polygons[(n_poly+n_new_polygons),(jj-kk)] = boundary_polygons[curr_poly,jj]
                    # slide everything in the current polygon backwards (or delete it)
                    for jj in range(j+1,n_edges+1):
                        boundary_polygons[curr_poly,jj-j+kk-1] =  boundary_polygons[curr_poly,jj]
                    for jj in range(n_edges-j+kk,n_edges+1):
                        boundary_polygons[curr_poly,jj] = -1
                    #for jj in range(kk,j):
                    #    if (j+jj) > n_edges:
                    #        boundary_polygons[curr_poly,jj] = -1
                    #    else:
                    #        boundary_polygons[curr_poly,jj] = boundary_polygons[curr_poly,(j+jj)]
                    #        boundary_polygons[curr_poly,(j+jj)] = -1  # restore to -1
                    print(f"{boundary_polygons[curr_poly,:14]}")
                    print(f"{boundary_polygons[(n_poly+n_new_polygons),:14]}")
                    # we moved everything backwards so we need to re-check this slot, which now contains a new value
                    # j -= 1

                    # decrement the number of pinches (there might be more than one, picture two triangluar boundaries
                    # coming off the same corner of a square boundary)        
                    self._cvertices[curr_edge.vertex].component -= 1
                    _orig = -1
                else:
                    _orig = curr_edge.vertex  # now we use this to track vertex
                    kk = j+1  # use this to track starting index
            j += 1
            if j >= n_edges:
                break

        return n_new_polygons

    cdef _pinch_boundaries(self, np.ndarray boundary_polygons, bint live_update=1):
        """
        Take an array of boundary polygons (each row is a polygon) and pinch them to close.

        NOTE: This implementation seems to produce singularities.

        References
        ----------
        Gueziec et al. Cutting and Stitching: Converting Sets of Polygons
        to Manifold Surfaces, IEEE TRANSACTIONS ON VISUALIZATION AND 
        COMPUTER GRAPHICS, 2001.
        """
        cdef int j, k, n_edges, n_pinch
        cdef bint odd
        cdef np.int32_t _edge0, _edge1
        cdef int n_poly = boundary_polygons.shape[0]
        cdef int max_edges = boundary_polygons.shape[1]

        # Move down the row of polygons
        j = 0
        while (j < n_poly) and (boundary_polygons[j,0] != -1):
            #if j != 1:
            #    j += 1
            #    continue
            # Count the number of elements in this polygon
            n_edges = 0
            while (n_edges < max_edges) and (boundary_polygons[j,n_edges] != -1):
                n_edges += 1

            if n_edges < 2:
                print("ERROR: The absolute minimum polygon size is 2. Something's wrong. Skipping.")
                continue

            #if n_edges > 6:
            #    print(f"Skipping polygon {j} of size {n_edges}")
            #    j += 1
            #    continue

            #print(f"Polygon {j} of size {n_edges}")
            
            odd = ((n_edges % 2) != 0)

            if odd:
                # If the length of this polygon is odd, we pinch all but the last three edges...
                n_pinch = (n_edges-3)//2
            else:
                # ...otherwise we pinch every edge
                n_pinch = (n_edges)//2

            # Now pinch elements together
            #k = 2
            for k in range(n_pinch):
                # Run around the closed boundary loop in opposite directions
                _edge0 = boundary_polygons[j,k]
                _edge1 = boundary_polygons[j,n_edges-k-1]
            
                self._pinch_edges(_edge0, _edge1, live_update=live_update)

            if odd:
                #print("We went odd!")
                # Make a triangle from the last three edges to seal the boundary
                self._fill_triangle(boundary_polygons[j,n_pinch], boundary_polygons[j,n_pinch+1], boundary_polygons[j,n_pinch+2], live_update=live_update)

            j += 1
        
        self._invalidate_cached_properties()

    cdef _pinch_edges(self, np.int32_t _edge0, np.int32_t _edge1, bint live_update=1):
        """
        Pinch two (boundary) edges together to stitch part of the mesh.
        """
        cdef int i
        cdef np.int32_t _vertex0, _vertex1
        cdef vertex_d *vertex0
        cdef vertex_d *vertex1
        cdef float pos0, pos1, pos2
        cdef halfedge_t *edge0
        cdef halfedge_t *edge1

        cdef int n_halfedges = self._halfedges.shape[0]
        
        edge0 = &self._chalfedges[_edge0]
        edge1 = &self._chalfedges[_edge1]

        if (edge0.twin != -1) or (edge1.twin != -1):
            print("WARNING: One of these is not a boundary!!! Are you sure you want to stitch them?")

        if (edge1.vertex != self._chalfedges[edge0.prev].vertex):
            print("WARNING: These are not continuous!")

        # update pinched vertex position
        _vertex0 = edge0.vertex
        _vertex1 = self._chalfedges[edge1.prev].vertex

        #print(f"Check vertices: {_vertex1} {edge1.vertex} {self._chalfedges[edge0.prev].vertex} {_vertex0}")
        #print("Pre-stitch")
        #print(edge0.vertex, edge0.face, edge0.twin, edge0.next, edge0.prev, edge0.length, edge0.component, edge0.locally_manifold)
        #print(edge1.vertex, edge1.face, edge1.twin, edge1.next, edge1.prev, edge1.length, edge1.component, edge1.locally_manifold)

        if _vertex0 != _vertex1:
            vertex0 = &self._cvertices[_vertex0]
            vertex1 = &self._cvertices[_vertex1]
            pos0 = 0.5*(vertex0.position0 + vertex1.position0)
            pos1 = 0.5*(vertex0.position1 + vertex1.position1)
            pos2 = 0.5*(vertex0.position2 + vertex1.position2)
            vertex0.position0 = pos0
            vertex0.position1 = pos1
            vertex0.position2 = pos2

            # set all vertex0 points to vertex1
            for i in range(n_halfedges):
                if self._chalfedges[i].vertex == _vertex1:
                    self._chalfedges[i].vertex = _vertex0

            # Delete this vertex
            self._vertex_delete(_vertex1)
            
        # update connectivity
        edge0.twin = _edge1
        edge1.twin = _edge0

        if live_update:
            update_face_normal(int(edge0.face), self._chalfedges, self._cvertices, self._cfaces)
            update_face_normal(int(edge1.face), self._chalfedges, self._cvertices, self._cfaces)

            update_single_vertex_neighbours(int(edge0.vertex), self._chalfedges, self._cvertices, self._cfaces)
            update_single_vertex_neighbours(int(edge1.vertex), self._chalfedges, self._cvertices, self._cfaces)

            #print("Post-stitch")
            #print(edge0.vertex, edge0.face, edge0.twin, edge0.next, edge0.prev, edge0.length, edge0.component, edge0.locally_manifold)
            #print(edge1.vertex, edge1.face, edge1.twin, edge1.next, edge1.prev, edge1.length, edge1.component, edge1.locally_manifold)
            self._singular_edges_valid = 0
            #print(self.singular_edges)

            #for i in range(self.singular_edges.shape[0]):
            #    edge0 = &self._chalfedges[self.singular_edges[i]]
            #    print(edge0.vertex, edge0.face, edge0.twin, edge0.next, edge0.prev, edge0.length, edge0.component, edge0.locally_manifold)

    cdef _color_boundaries(self, np.ndarray boundary_polygons):
        """
        Color each boundary a single color.
        """
        cdef int j, n_edges

        self._components_valid = 1

        # zero-out components
        for j in range(self._vertices.shape[0]):
            self._cvertices[j].component = -1

        # Move down the row of polygons
        j = 0
        while boundary_polygons[j,0] != -1:
            # Count the number of elements in this polygon
            n_edges = 0
            #if j != 1:
            #    j+=1
            #    continue
            while boundary_polygons[j,n_edges] != -1:
                #self._cvertices[self._chalfedges[self._chalfedges[boundary_polygons[j,n_edges]].prev].vertex].component
                self._cvertices[self._chalfedges[boundary_polygons[j,n_edges]].vertex].component = j # n_edges
                #self._cvertices[self._chalfedges[boundary_polygons[j,n_edges]].vertex].component = n_edges
                n_edges += 1
            j += 1

    def _fill_triangle(self, np.int32_t h0, np.int32_t h1, np.int32_t h2, bint live_update=1):
        """
        Make a triangle from the three open edges to seal a boundary
        """
        cdef halfedge_t *edge0
        cdef halfedge_t *edge1
        cdef halfedge_t *edge2

        n_edges = self.new_edges(3)
        n_faces = self.new_faces(1)

        edge0 = &self._chalfedges[h0]
        edge1 = &self._chalfedges[h1]
        edge2 = &self._chalfedges[h2]

        self._populate_edge(n_edges[0], edge2.vertex, prev=n_edges[1], next=n_edges[2], face=n_faces[0], twin=h0)
        self._populate_edge(n_edges[1], edge0.vertex, prev=n_edges[2], next=n_edges[0], face=n_faces[0], twin=h1)
        self._populate_edge(n_edges[2], edge1.vertex, prev=n_edges[0], next=n_edges[1], face=n_faces[0], twin=h2)

        self._chalfedges[h0].twin = n_edges[0]
        self._chalfedges[h1].twin = n_edges[1]
        self._chalfedges[h2].twin = n_edges[2]

        self._cfaces[n_faces[0]].halfedge = n_edges[0]

        if live_update:
            update_face_normal(int(n_faces[0]), self._chalfedges, self._cvertices, self._cfaces)
                        
            update_single_vertex_neighbours(int(edge2.vertex), self._chalfedges, self._cvertices, self._cfaces)
            update_single_vertex_neighbours(int(edge1.vertex), self._chalfedges, self._cvertices, self._cfaces)
            update_single_vertex_neighbours(int(edge0.vertex), self._chalfedges, self._cvertices, self._cfaces)

    cdef _zig_zag_triangulation(self, np.ndarray boundary_polygons, np.int32_t *n_edges, np.int32_t *n_faces, int row, int len_poly, bint live_update=1):
        """
        Triangulate a (boundary) polygon (an array of halfedges) by zig-zagging 
        between its edges.
        """
        cdef int k
        cdef int _edge0, _edge1
        cdef halfedge_t *edge0
        cdef halfedge_t *edge1
        cdef bint zig

        # How many edges are in this polygon?
        # len_poly = 0
        # while poly[len_poly] != -1:
        #     print(poly[len_poly])
        #     len_poly += 1

        if len_poly < 3:
            print("ERROR: Cannot triangulate polygons with length less than 3. Aborting.")

        # starting edges
        _edge0 = boundary_polygons[row,0]
        _edge1 = boundary_polygons[row,1]
        
        # Zig-zag with the new edges
        zig = 0
        for k in range(len_poly-2):
            #print(k, _edge0, _edge1)
            # grab the two existing edges on the polygon
            edge0 = &self._chalfedges[_edge0]
            edge1 = &self._chalfedges[_edge1]

            #print(edge0.vertex, edge0.face, edge0.twin, edge0.next, edge0.prev, edge0.length, edge0.component, edge0.locally_manifold)
            #print(edge1.vertex, edge1.face, edge1.twin, edge1.next, edge1.prev, edge1.length, edge1.component, edge1.locally_manifold)

            # Populate the new triangle
            self._populate_edge(n_edges[3*k], self._chalfedges[edge0.prev].vertex, prev=n_edges[3*k+1], next=n_edges[3*k+2], face=n_faces[k], twin=_edge0)
            self._populate_edge(n_edges[3*k+1], edge0.vertex, prev=n_edges[3*k+2], next=n_edges[3*k], face=n_faces[k], twin=_edge1)
            self._populate_edge(n_edges[3*k+2], edge1.vertex, prev=n_edges[3*k], next=n_edges[3*k+1], face=n_faces[k], twin=-1)
            self._cfaces[n_faces[k]].halfedge = n_edges[3*k]

            # stitch to previously-existing edges
            self._chalfedges[_edge0].twin = n_edges[3*k]
            self._chalfedges[_edge1].twin = n_edges[3*k+1]

            if live_update:
                update_face_normal(int(n_faces[k]), self._chalfedges, self._cvertices, self._cfaces)
                            
                update_single_vertex_neighbours(int(self._chalfedges[edge0.prev].vertex), self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(int(edge0.vertex), self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(int(edge1.vertex), self._chalfedges, self._cvertices, self._cfaces)

            if k >= (len_poly-3):
                # don't index beyond the length of new_edge
                break
    
            # Choose the next two edges to stitch
            if zig:
                _edge0 = n_edges[3*k+2]
                _edge1 = boundary_polygons[row,(k+1)//2+1]
            else:
                _edge0 = boundary_polygons[row,len_poly-k//2-1]
                _edge1 = n_edges[3*k+2]
            zig = 1-zig

        # stitch the last edge
        _edge0 = boundary_polygons[row,len_poly-(len_poly-2)//2-1]
        _edge1 = n_edges[3*(len_poly-3)+2]
        self._chalfedges[_edge0].twin = _edge1
        self._chalfedges[_edge1].twin = _edge0

        if live_update:                        
            update_single_vertex_neighbours(int(self._chalfedges[_edge0].vertex), self._chalfedges, self._cvertices, self._cfaces)
            update_single_vertex_neighbours(int(self._chalfedges[_edge1].vertex), self._chalfedges, self._cvertices, self._cfaces)

    def _fill_holes(self, np.ndarray boundary_polygons, bint live_update=1):
        """
        Fill in the mesh holes: closed loops of boundary edges, one per row in boundary_polygons.
        """
        cdef int j, n_edges
        cdef int n_poly = boundary_polygons.shape[0]
        cdef int max_edges = boundary_polygons.shape[1]

        j = 0
        while (j < n_poly) and (boundary_polygons[j,0] != -1):
            # Count the number of elements in this polygon
            n_edges = 0
            while (n_edges < max_edges) and (boundary_polygons[j,n_edges] != -1):
                n_edges += 1

            #print(f"Polygon {j} of length {n_edges}")

            if n_edges == 2:
                self._pinch_edges(boundary_polygons[j,0], boundary_polygons[j,1], live_update=live_update)
            else:
                # Initialize new edges/faces
                new_faces = self.new_faces(int(n_edges-2))
                new_edges = self.new_edges(int(3*(n_edges-3)+3))

                # Right now we use zig-zag, but we should use Liepa, Peter. "Filling Holes in Meshes." 
                # In Proceedings of the 2003 Eurographics/ACM SIGGRAPH Symposium on Geometry Processing, 
                # 200-205. SGP '03. Goslar, DEU: Eurographics Association, 2003.

                self._zig_zag_triangulation(boundary_polygons, 
                                            <np.int32_t *> np.PyArray_DATA(new_edges), 
                                            <np.int32_t *> np.PyArray_DATA(new_faces), 
                                            j, n_edges, live_update=live_update)

            j += 1

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
        cdef int i, j, k, component, n_v
        cdef np.int32_t twin_idx, next_idx

        # Iterate over vertices that are endpoints of a singular edge or are
        # isolated singular vertices (number of incident edges =/= number of
        # neighbors)
        self._singular_vertices_valid = 0
        self._singular_edges_valid = 0
        singular_vertices = self.singular_vertices

        for i in range(singular_vertices.shape[0]):
            # Find the star of this vertex. That is, all of the vertices
            # edges incident on this vertex. We grab only the edges pointing at the vertex
            # because 1) this is sufficient and 2) this allows unambiguous traversal of
            # the one-ring neighbor via twin->next movement 
            star = np.flatnonzero(self._halfedges['vertex'] == singular_vertices[i]).astype('i4')

            # Partition the faces of the star s.t. two faces are connected if 
            # they share a non-singular edge. Denote the number of connected 
            # components under this relation as component.
            for j in range(star.shape[0]):
                self._chalfedges[star[j]].component = -1
                self._chalfedges[self._chalfedges[star[j]].next].component = -1
                self._chalfedges[self._chalfedges[star[j]].prev].component = -1
            component = 0
            for j in range(star.shape[0]):
                component = flood_fill_star_component(star[j], component, self._chalfedges)

            # Construct n_v = component-1 additional compies of vertex i...
            n_v = component-1
            if n_v < 1:
                continue
            n_vertices = self.new_vertices(n_v)
            for k in range(n_v):
                # ...with the same coordinates
                self._cvertices[n_vertices[k]].position0 = copy.copy(self._cvertices[singular_vertices[i]].position0) #+ 10*(k+1)*self._cvertices[singular_vertices[i]].normal0
                self._cvertices[n_vertices[k]].position1 = copy.copy(self._cvertices[singular_vertices[i]].position1) #+ 10*(k+1)*self._cvertices[singular_vertices[i]].normal1
                self._cvertices[n_vertices[k]].position2 = copy.copy(self._cvertices[singular_vertices[i]].position2) #+ 10*(k+1)*self._cvertices[singular_vertices[i]].normal2
                self._cvertices[n_vertices[k]].halfedge = -1

            # Loop over the star again, assigning vertices to each component
            for j in range(star.shape[0]):
                k = self._chalfedges[star[j]].component
                if k < n_v:
                    # Assign halfedges associated with component k \in range(n_v) a unique copy of this vertex
                    self._chalfedges[star[j]].vertex = n_vertices[k]

                # Mark singular edges in this component as non-singular, boundary
                if self._chalfedges[star[j]].locally_manifold == 0:
                    #print(f"Singular in component {k}")
                    #self._chalfedges[star[j]].locally_manifold = 1
                    twin_idx = self._chalfedges[star[j]].twin
                    if twin_idx != -1:
                        self._chalfedges[twin_idx].twin = -1
                        self._chalfedges[star[j]].twin = -1

                next_idx = self._chalfedges[star[j]].next
                if self._chalfedges[next_idx].locally_manifold == 0:
                    #print(f"Next singular in component {k}")
                    #self._chalfedges[next_idx].locally_manifold = 1
                    twin_idx = self._chalfedges[next_idx].twin
                    if twin_idx != -1:
                        self._chalfedges[twin_idx].twin = -1
                        self._chalfedges[next_idx].twin = -1

                # find an emanating halfedge in component k
                self._cvertices[self._chalfedges[star[j]].vertex].halfedge = next_idx

            for j in range(star.shape[0]):
                update_face_normal(int(self._chalfedges[star[j]].face), self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(int(self._chalfedges[star[j]].vertex), self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(int(self._chalfedges[self._chalfedges[star[j]].prev].vertex), self._chalfedges, self._cvertices, self._cfaces)
                update_single_vertex_neighbours(int(self._chalfedges[self._chalfedges[star[j]].next].vertex), self._chalfedges, self._cvertices, self._cfaces)

        # reset
        self._invalidate_cached_properties()

    def repair(self, bint only_largest_component=0, bint pinch=0):
        """
        Repair the mesh so it's topologically manifold.

        References
        ----------
        M. Attene, A lightweight approach to repairing digitized polygon
        meshes, The Visual Computer, 2010.
        """
        
        # 1. Remove singularities (singular edges and isolated singular vertices)
        self._remove_singularities()

        # (Optional) 2. Remove all connected components except the largest
        if only_largest_component:
            self.keep_largest_connected_component()

        #self.remesh(n=1)

        # construct an array to store the boundary polygons
        # the longest boundary polygon possible contains all of the boundary edges in the mesh,
        # the shortest boundary "polygon" possible is of length 2
        # TODO: This could be quite large if we have a lot of boundary edges
        boundary_edges = np.flatnonzero((self._halfedges['vertex'] != -1) & (self._halfedges['twin'] == -1)).astype('i4')
        max_poly_length = max(boundary_edges.shape[0],4)
        boundary_polygons = -1*np.ones([max_poly_length//2, max_poly_length], dtype='i4')

        # 3. Patch mesh holes with new triangles
        self._find_boundary_polygons(boundary_polygons, boundary_edges)  # fill the boundary polygon array

        if pinch == 1:
            self._pinch_boundaries(boundary_polygons)                        # close the boundary polygons by pinching 
        else:
            self._fill_holes(boundary_polygons)
        # self._color_boundaries(boundary_polygons)

        # Reset
        #self._manifold = None
        # self._faces['normal'][:] = -1
        # self._vertices['normal'][:] = -1
        #self._face_normals_valid = 0
        #self._vertex_normals_valid = 0
        self._invalidate_cached_properties()
        self._update_all_face_normals()
        self._update_all_vertex_neighbours()

    def deneck(self, e_threshold=30.0):
        """
        Remove topological necks by deleting high energy vertices and patching the holes
        this creates.

        TODO - threshold is very emperical.
        TODO - is E actually defined here, or only in subclasses?
        """
        verts = np.flatnonzero(self.E > e_threshold)
        self.unsafe_remove_vertices(verts)
        self.repair()
    
    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    def unsafe_remove_vertices(self, ids_to_remove):
        """
        Remove a vertex and incident halfedges/faces.

        NOTE: this will leave a non-manifold mesh! We then rely on repair to tidy it up.
        """

        #ids_to_remove = [id,]

        cdef int id, h, f, i, hvx, cand, ht, j
        cdef int n_vertices = self._vertices.shape[0]

        cdef int [::1]  _h_to_remove
        cdef int _n_h_to_remove, c_in_h_to_remove

        assert(len(ids_to_remove) > 0)

        h0 = np.concatenate([np.argwhere(self._halfedges['vertex']==id).squeeze() for id in ids_to_remove])
        h1 = self._halfedges['twin'][h0]
        h1 = h1[h1 != -1]

        h_to_remove = np.concatenate((h0, h1))

        #print([self._halfedges['face'][h] for h in h_to_remove])

        f_to_remove = np.unique([f for f in [self._halfedges['face'][h] for h in h_to_remove] if f >=0])

        h_to_remove = np.array([h for h in np.unique(np.concatenate([np.argwhere(self._halfedges['face'] == f).squeeze() for f in f_to_remove])) if h >=0], 'i')

        _h_to_remove = h_to_remove
        _n_h_to_remove = len(h_to_remove)
        
        # make sure that none of the halfedges that we remove is referenced from a vertex
        # that should remain
        for j in range(_n_h_to_remove):
            h = _h_to_remove[j]
            #try:
            #    hvx = np.flatnonzero(self._vertices['halfedge'] == h)[0]
            #except IndexError:
            #    # no vertices point to this halfedge - we don't need to do anything
            #    continue

            hvx = -1
            for i in range(n_vertices):
                if self._cvertices[i].halfedge == h:
                    hvx = i
                    break

            if hvx == -1:
                # we did not find any vertex which points to this half edge
                continue 


            if hvx in ids_to_remove:
                # we are going to remove the vertex, ignore
                continue

            #iterate around the vertex trying to find a halfedge that will remain
            cand = self._chalfedges[self._chalfedges[h].twin].next
            
            # cand in h_to_remove
            c_in_h_to_remove = 0
            for i in range(_n_h_to_remove):
                c_in_h_to_remove |= (cand == _h_to_remove[i])

            while (cand != h) and (cand != -1) and c_in_h_to_remove:
                cand = self._chalfedges[self._chalfedges[cand].twin].next

                # cand in h_to_remove
                c_in_h_to_remove = 0
                for i in range(_n_h_to_remove):
                    c_in_h_to_remove |= (cand == _h_to_remove[i])

            # didn't find a candidate, iterate in other direction
            if (cand != h) or (cand != -1) or (c_in_h_to_remove):
                cand = self._chalfedges[self._chalfedges[h].prev].twin

                # cand in h_to_remove
                c_in_h_to_remove = 0
                for i in range(_n_h_to_remove):
                    c_in_h_to_remove |= (cand == _h_to_remove[i])

                while (cand != h) and (cand != -1) and (c_in_h_to_remove):
                    cand = self._chalfedges[self._chalfedges[cand].prev].twin

                    # cand in h_to_remove
                    c_in_h_to_remove = 0
                    for i in range(_n_h_to_remove):
                        c_in_h_to_remove |= (cand == _h_to_remove[i])
        
            if (cand == h) or (cand == -1) or (c_in_h_to_remove):
                
                #print(f'h: {h}, {self._halfedges[h]}')
                # Check if this is a single vertex on it's own
                vtx_halfedges = np.argwhere(self._halfedges['vertex']==hvx).squeeze()
                if np.all([vtx_h in h_to_remove for vtx_h in vtx_halfedges]):
                    print('All incident halfedges are being removed, remove vertex too')
                    ids_to_remove = np.concatenate([ids_to_remove, [hvx, ]])
                else:
                    print('ERROR: could not find a suitable replacement vertex halfedge, mesh will be singular')
            else:
                self._cvertices[hvx].halfedge = cand      
            
        
        # remove twin references
        for h in h_to_remove:
            ht = self._chalfedges[h].twin
            if ht >=0:
                self._chalfedges[ht].twin = -1

        self._halfedges[h_to_remove] = -1
        self._halfedge_vacancies.extend(h_to_remove)

        self._vertices[ids_to_remove] = -1
        self._vertex_vacancies.extend(ids_to_remove)

        self._faces[f_to_remove] = -1
        self._face_vacancies.extend(f_to_remove)



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

        triangles_stl = np.zeros(self.faces.shape[0], dtype=dt)
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
        vertex_lookup = np.zeros(self._vertices.shape[0], dtype=np.int)
        
        vertex_lookup[live_vertices] = new_vertex_indices

        # Grab the faces and vertices we want
        faces = vertex_lookup[self.faces]
        vertices = self._vertices['position'][live_vertices]

        ply.save_ply(filename, vertices, faces, colors)
        
    def plot(self, figure=None, axis=None):
        """
        Display surface using matplotlib. Useful for quick use in a notebook (or similar) when a wx GUI is not
        available. NOTE: will be slower than displaying through PYMEVis

        Parameters
        ----------
        figure : [optional] a matplotlib figure instance
        axis : [optional] a matplotlib Axes3D instance

        Returns
        -------

        """
        import matplotlib.pyplot as plt

        if axis is None:
            if figure is None:
                figure = plt.figure()

            axis = figure.add_subplot(projection='3d')

        return axis.plot_trisurf(self.vertices[:,0], self.vertices[:,1], self.faces, self.vertices[:,2])
