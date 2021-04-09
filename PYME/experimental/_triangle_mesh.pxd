cimport numpy as np

cdef extern from 'triangle_mesh_utils.h':
    const int NEIGHBORSIZE  # Note this must match NEIGHBORSIZE in triangle_mesh_utils.h
    const int VECTORSIZE
    
    cdef struct halfedge_t:
        np.int32_t vertex
        np.int32_t face
        np.int32_t twin
        np.int32_t next
        np.int32_t prev
        np.float32_t length
        np.int32_t component
        
    cdef struct face_t:
        np.int32_t halfedge
        float normal[VECTORSIZE]
        float area
        np.int32_t component
        
    cdef struct face_d:
        np.int32_t halfedge
        np.float32_t normal0
        np.float32_t normal1
        np.float32_t normal2
        np.float32_t area
        np.int32_t component
        
    cdef struct vertex_t:
        float position[VECTORSIZE];
        float normal[VECTORSIZE];
        np.int32_t halfedge;
        np.int32_t valence;
        np.int32_t neighbors[NEIGHBORSIZE];
        np.int32_t component;
        np.int32_t locally_manifold
        
    cdef struct vertex_d:
        np.float32_t position0
        np.float32_t position1
        np.float32_t position2
        np.float32_t normal0
        np.float32_t normal1
        np.float32_t normal2
        np.int32_t halfedge
        np.int32_t valence
        np.int32_t neighbor0
        np.int32_t neighbor1
        np.int32_t neighbor2
        np.int32_t neighbor3
        np.int32_t neighbor4
        np.int32_t neighbor5
        np.int32_t neighbor6
        np.int32_t neighbor7
        np.int32_t neighbor8
        np.int32_t neighbor9
        np.int32_t neighbor10
        np.int32_t neighbor11
        np.int32_t neighbor12
        np.int32_t neighbor13
        np.int32_t neighbor14
        np.int32_t neighbor15
        np.int32_t neighbor16
        np.int32_t neighbor17
        np.int32_t neighbor18
        np.int32_t neighbor19
        np.int32_t component
        np.int32_t locally_manifold

cdef class TrianglesBase(object):
    pass

cdef class TriangleMesh(TrianglesBase):
    cdef public object _vertices
    cdef public object _faces
    cdef public object _halfedges
    cdef public object _vertex_vacancies
    cdef public object _face_vacancies
    cdef public object _halfedge_vacancies

    cdef halfedge_t * _chalfedges
    cdef face_d * _cfaces
    cdef vertex_d * _cvertices

    cdef object _faces_by_vertex
    cdef object _loop_subdivision_flip_edges
    cdef object _loop_subdivision_new_vertices

    cdef object _singular_edges
    cdef object _singular_vertices

    cdef public object vertex_properties
    cdef public object extra_vertex_data
    cdef object fix_boundary
    cdef object _manifold

    cdef object _H
    cdef object _K
    cdef public object smooth_curvature

    cdef _set_chalfedges(self, halfedge_t[:])
    cdef _set_cfaces(self, face_d[:])
    cdef _set_cvertices(self, vertex_d[:])
    cdef bint _check_neighbour_twins(self, int)
    cdef bint _check_collapse_fast(self, int, int)
    cdef bint _check_collapse_slow(self, int, int)
    cdef _face_delete(self, np.int32_t)
    cdef _edge_delete(self, np.int32_t)
    cdef _zipper(self, np.int32_t, np.int32_t)
    cpdef int edge_flip(self, np.int32_t, bint live_update=*)
