cimport cython
cimport numpy as np
from cpython cimport PyObject_CheckBuffer, PyObject_GetBuffer, PyBuffer_Release, Py_buffer, PyObject, PyBUF_SIMPLE, PyBUF_C_CONTIGUOUS, PyBUF_F_CONTIGUOUS, PyBuffer_IsContiguous

import triangle_mesh 

# cdef packed struct halfedge_t:
#     np.int32_t vertex
#     np.int32_t face
#     np.int32_t twin
#     np.int32_t next
#     np.int32_t prev
#     np.int32_t length

# cdef packed struct face_t:
#     np.int32_t halfedge
#     np.float32_t[3] normal
#     np.float32_t area

# cdef packed struct vertex_t:
#     np.float32_t[3] position
#     np.float32_t[3] normal
#     np.int32_t halfedge
#     np.int32_t valence
#     np.int32_t[8] neighbors

cdef extern from "triangle_mesh_utils.h":
    ctypedef struct halfedge_t:
        pass
    ctypedef struct vertex_t:
        pass
    ctypedef struct face_t:
        pass
    void update_vertex_neighbors(np.int32_t *v_idxs, halfedge_t *halfedges, vertex_t *vertices, face_t *faces, np.int32_t n_idxs)

@cython.boundscheck(False)
@cython.wraparound(False)
def c_update_vertex_neighbors(mesh, v_idxs):

    cdef np.int32_t n_v_idxs = len(v_idxs)
    
    cdef Py_buffer v_idxs_buf
    cdef Py_buffer halfedges_buf
    cdef Py_buffer vertices_buf
    cdef Py_buffer faces_buf

    PyObject_GetBuffer(v_idxs, &v_idxs_buf, PyBUF_C_CONTIGUOUS)
    PyObject_GetBuffer(mesh._halfedges, &halfedges_buf, PyBUF_C_CONTIGUOUS)
    PyObject_GetBuffer(mesh._vertices, &vertices_buf, PyBUF_C_CONTIGUOUS)
    PyObject_GetBuffer(mesh._faces, &faces_buf, PyBUF_C_CONTIGUOUS)

    update_vertex_neighbors(<np.int32_t *> v_idxs_buf.buf, <halfedge_t *> halfedges_buf.buf, <vertex_t *> vertices_buf.buf, <face_t *> faces_buf.buf, n_v_idxs)

    PyBuffer_Release(&v_idxs_buf)
    PyBuffer_Release(&halfedges_buf)
    PyBuffer_Release(&vertices_buf)
    PyBuffer_Release(&faces_buf)
