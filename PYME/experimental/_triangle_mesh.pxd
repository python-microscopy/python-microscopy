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
        np.int32_t locally_manifold
        
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

    # cdef object _singular_edges
    # cdef object _singular_vertices
    cdef bint _singular_edges_valid
    cdef bint _singular_vertices_valid

    cdef public object vertex_properties
    cdef public object extra_vertex_data
    cdef object fix_boundary
    cdef object _manifold

    cdef bint _vertex_normals_valid
    cdef bint _face_normals_valid

    cdef bint _components_valid

    cdef object _H
    cdef object _K
    cdef public object smooth_curvature

    cdef _set_chalfedges(self, np.ndarray)#halfedge_t[:])
    cdef _set_cfaces(self, np.ndarray)#face_d[:])
    cdef _set_cvertices(self, np.ndarray)#vertex_d[:])
    cdef bint _check_neighbour_twins(self, int)
    cdef bint _check_collapse_fast(self, int, int)
    cdef bint _check_collapse_slow(self, int, int)
    cdef int _face_delete(self, np.int32_t)
    cdef int _edge_delete(self, np.int32_t)
    cdef int _vertex_delete(self, np.int32_t v_idx)
    cdef _zipper(self, np.int32_t, np.int32_t)
    cpdef int edge_flip(self, np.int32_t, bint live_update=*)
    cpdef int edge_collapse(self, np.int32_t, bint live_update=*)
    cpdef int edge_split(self, np.int32_t, bint live_update=*, bint upsample=*)
    cpdef int relax(self, float l=*, int n=*)
    cpdef int regularize(self)

    cdef int edge_split_2(self, np.int32_t _curr, np.int32_t* new_edges, np.int32_t* new_vertices, np.int32_t* new_faces, int n_edge_idx, int n_vertex_idx, int n_face_idx,  
                            bint live_update=*, bint upsample=*)
    
    cdef _populate_edge(self, int idx, int vertex, int prev=*, int next=*, int face=*, int twin=*, int locally_manifold=*)
    #cdef int _insert_new_edge(self, int vertex, int prev=-1, int next=-1, int face=-1, int twin=-1
    
    cdef int _new_face(self, int)
    #cdef int _new_vertex(self, np.ndarray, int)

    cdef _compute_raw_vertex_valences(self, np.ndarray valence)

    cdef _find_boundary_polygons(self, np.ndarray boundary_polygons, np.ndarray boundary_edges)
    cdef int _disconnect_pinched_polygons(self, np.ndarray boundary_polygons, int curr_poly, int n_poly, int n_edges)
    cdef _pinch_boundaries(self, np.ndarray boundary_polygons, bint live_update=*)
    cdef _pinch_edges(self, np.int32_t _edge0, np.int32_t _edge1, bint live_update=*)
    cdef _color_boundaries(self, np.ndarray boundary_polygons)
    cdef _zig_zag_triangulation(self, np.ndarray boundary_polygons, np.int32_t *n_edges, np.int32_t *n_faces, int row, int len_poly, bint live_update=*)
