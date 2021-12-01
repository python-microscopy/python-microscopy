// #include <stdio.h>
// #include <stdlib.h>
#include "Python.h"
#include <math.h>
#include "numpy/arrayobject.h"

#include "triangle_mesh_utils.h"

#define EPS 1e-12

float norm(const float *pos)
{
    float n = 0;
    int i = 0;

    for (i = 0; i < VECTORSIZE; ++i)
        n += pos[i] * pos[i];
    return sqrt(n);
}

void cross(const float *a, const float *b, float *n)
{
    /*
    One thing that might be worth thinking about is making a typedef for a vector
    - As it stands, the compiler doesn't have any info about the overall dimensions 
    of a, b, or n, or their alignment in memory but I'm not sure that matters.
    */

    n[0] = a[1]*b[2] - a[2]*b[1];
    n[1] = a[2]*b[0] - a[0]*b[2];
    n[2] = a[0]*b[1] - a[1]*b[0];
}

void difference(const float *a, const float *b, float *d)
{
    int k = 0;
    for (k=0; k < VECTORSIZE; ++k)
        d[k] = a[k] - b[k];
}

// void update_vertex_neighbors(signed int *v_idxs, halfedge_t *halfedges, vertex_t *vertices, face_t *faces, signed int n_idxs)

/*
DB: Performance suggestions:

- change to a pure c function and a python c api wrapper
  reasons are 1 : facilitates moving more of the code to c 2: forces us to code in a way that doesn't make python API
  calls within loop. This is good both because it will result in better performance, and also in that it will (potentially)
  let us release the GIL and get performance improvements from multithreading

- PySequence_GetItem() is a python API call, we probably don't want it in the loop if we can help it.

- I'm pretty sure that PyArray_GETPTR1 is a macro, which makes it significantly less costly (expands to inline c code, &
  no need for GIL). That said, we know that the input arrays are contiguous (we check for this) so we could also just have,
  e.g., vertex_t * p_vertices = PyArray_GETPTR1(vertices, 0) outside the loop and then just replace everything inside
  the loop with standard c indexing - ie curr_vertex = p_vertices[v_idx]. My strong suspicion is that the c compiler will do a
  better job of optimizing this, and it will also continue to work once more of the code is in c.

*/

static void update_single_vertex_neighbours(int v_idx, halfedge_t *halfedges, void *vertices_, void *faces_)
{
    int32_t i, k, orig_idx, curr_idx, twin_idx, tmp;
    halfedge_t *curr_edge, *twin_edge;
    vertex_t *curr_vertex, *loop_vertex;
    face_t *curr_face;

    float position[VECTORSIZE];
    float normal[VECTORSIZE];
    float a, l, nn;

    vertex_t *vertices = (vertex_t*) vertices_;
    face_t *faces = (face_t*) faces_;

    curr_vertex = &(vertices[v_idx]);

    curr_idx = curr_vertex->halfedge;
    orig_idx = curr_idx;

    if (curr_idx == -1) return;

    curr_edge = &(halfedges[curr_idx]);
    twin_idx = curr_edge->twin;
    if (twin_idx != -1)
        twin_edge = &(halfedges[twin_idx]);

    i = 0;

    //zero out neighbours and normal
    for (k = 0; k < NEIGHBORSIZE; ++k)
        (curr_vertex->neighbors)[k] = -1;

    for (k = 0; k < VECTORSIZE; ++k)
        normal[k] = 0;

    while (1)
    {
        if (curr_idx == -1)
            break;

        // printf("Current index: %d\n", curr_idx);
        // printf("orig_index: %d\n", orig_idx);

        if (i < NEIGHBORSIZE)
        {
            (curr_vertex->neighbors)[i] = curr_idx;
            curr_face = &(faces[(curr_edge->face)]);
            a = curr_face->area;
            for (k = 0; k < VECTORSIZE; ++k)
                normal[k] += ((curr_face->normal)[k])*a;
        }

        loop_vertex = &(vertices[(curr_edge->vertex)]);

        difference((curr_vertex->position), (loop_vertex->position), position);

        l = norm(position);
        curr_edge->length = l;
        if (twin_idx == -1)
            break;
        twin_edge->length = l;

        curr_idx = twin_edge->next;
        curr_edge = &(halfedges[curr_idx]);
        twin_idx = curr_edge->twin;
        twin_edge = &(halfedges[twin_idx]);

        ++i;

        if (curr_idx == orig_idx)
            break;
    }

    // If we hit a boundary, try the reverse direction,
    // starting from orig_idx
    // twin now becomes prev
    if ((twin_idx == -1) && (curr_idx != -1) && (i < NEIGHBORSIZE))
    {
        // Ideally we sweep through the neighbors in a continuous fashion,
        // so we'll reverse the order of all the edges so far.
        for (k=i;k>1;k--)
        {
            tmp = (curr_vertex->neighbors)[k];
            (curr_vertex->neighbors)[k] = (curr_vertex->neighbors)[k-1];
            (curr_vertex->neighbors)[k-1] = tmp;
        }

        curr_idx = orig_idx;
        curr_edge = &(halfedges[curr_idx]);
        twin_idx = curr_edge->prev;
        if (twin_idx == -1)
            return;

        twin_edge = &(halfedges[twin_idx]);
        curr_idx = twin_edge->twin;
        if (curr_idx == -1)
            return;
        curr_edge = &(halfedges[curr_idx]);

        ++i;

        while (1)
        {
            if (curr_idx == -1)
                break;

            if (i < NEIGHBORSIZE)
            {
                (curr_vertex->neighbors)[i] = curr_idx;
                curr_face = &(faces[(curr_edge->face)]);
                a = curr_face->area;
                for (k = 0; k < VECTORSIZE; ++k)
                    normal[k] += ((curr_face->normal)[k])*a;
            }

            loop_vertex = &(vertices[(curr_edge->vertex)]);

            difference((curr_vertex->position), (loop_vertex->position), position);

            l = norm(position);
            curr_edge->length = l;
            twin_edge->length = l;

            twin_idx = curr_edge->prev;
            if (twin_idx == -1)
                break;
            twin_edge = &(halfedges[twin_idx]);
            curr_idx = twin_edge->twin;
            if (curr_idx == -1)
                break;
            curr_edge = &(halfedges[curr_idx]);

            if (curr_idx == orig_idx)
                break;

            ++i;
        }
    }

    curr_vertex->valence = i;

    nn = norm(normal);
    if (nn > EPS) {
        for (k = 0; k < VECTORSIZE; ++k)
            (curr_vertex->normal)[k] = normal[k]/nn;
    } else {
        for (k = 0; k < VECTORSIZE; ++k)
            (curr_vertex->normal)[k] = 0;
    }
}

static PyObject *update_vertex_neighbors(PyObject *self, PyObject *args)
{
    PyObject *v_idxs=0, *halfedges=0, *vertices=0, *faces=0;
    int32_t j, v_idx, n_idxs;
    halfedge_t *p_halfedges;
    vertex_t *p_vertices;
    face_t *p_faces;

    //float position[VECTORSIZE];
    //float normal[VECTORSIZE];
    //float a, l, nn;

    if (!PyArg_ParseTuple(args, "OOOO", &v_idxs, &halfedges, &vertices, &faces)) return NULL;
    if (!PySequence_Check(v_idxs))
    {
        PyErr_Format(PyExc_RuntimeError, "expecting an sequence  eg ... (xvals, yvals) or (xvals, yvals, zvals)");
        return NULL;
    }
    if (!PyArray_Check(halfedges) || !PyArray_ISCONTIGUOUS(halfedges))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the edge data.");
        return NULL;
    }
    if (!PyArray_Check(vertices) || !PyArray_ISCONTIGUOUS(vertices)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the vertex data.");
        return NULL;
    }
    if (!PyArray_Check(faces) || !PyArray_ISCONTIGUOUS(faces)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the face data.");
        return NULL;
    } 

    n_idxs = (int32_t)PySequence_Length(v_idxs);
    p_vertices = (vertex_t*)PyArray_GETPTR1(vertices, 0);
    p_halfedges = (halfedge_t*)PyArray_GETPTR1(halfedges, 0);
    p_faces = (face_t*)PyArray_GETPTR1(faces, 0);

    for (j = 0; j < n_idxs; ++j)
    {
        v_idx = (int32_t)PyArray_DATA(PySequence_GetItem(v_idxs, (Py_ssize_t) j));
        if (v_idx == -1)
            continue;
                
        update_single_vertex_neighbours(v_idx, p_halfedges, p_vertices, p_faces);
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *update_all_vertex_neighbors(PyObject *self, PyObject *args)
{
    PyObject *halfedges=0, *vertices=0, *faces=0;
    int j, n_idxs;
    halfedge_t *p_halfedges;
    vertex_t *p_vertices;
    face_t *p_faces;

    n_idxs = 0;

    if (!PyArg_ParseTuple(args, "iOOO", &n_idxs, &halfedges, &vertices, &faces)) return NULL;
    if (!PyArray_Check(halfedges) || !PyArray_ISCONTIGUOUS(halfedges))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the edge data.");
        return NULL;
    }
    if (!PyArray_Check(vertices) || !PyArray_ISCONTIGUOUS(vertices)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the vertex data.");
        return NULL;
    }
    if (!PyArray_Check(faces) || !PyArray_ISCONTIGUOUS(faces)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the face data.");
        return NULL;
    } 

    p_vertices = (vertex_t*)PyArray_GETPTR1(vertices, 0);
    p_halfedges = (halfedge_t*)PyArray_GETPTR1(halfedges, 0);
    p_faces = (face_t*)PyArray_GETPTR1(faces, 0);

    for (j = 0; j < n_idxs; ++j)
    {
        if (p_vertices[j].halfedge == -1) continue;
        update_single_vertex_neighbours(j, p_halfedges, p_vertices, p_faces);
    }

    Py_INCREF(Py_None);
    return Py_None;
}


static void update_face_normal(int f_idx, halfedge_t *halfedges, void *vertices_, void *faces_)
{
    int k, curr_idx, prev_idx, next_idx;

    float v1[VECTORSIZE], u[VECTORSIZE], v[VECTORSIZE], n[VECTORSIZE], nn;

    vertex_t *vertices = (vertex_t*) vertices_;
    face_t *faces = (face_t*) faces_;

    halfedge_t *curr_edge, *prev_edge, *next_edge;
    face_t *curr_face;
    vertex_t *curr_vertex, *prev_vertex, *next_vertex;

    curr_face = &(faces[f_idx]);

    curr_idx = curr_face->halfedge;
    if (curr_idx == -1) return;

    curr_edge = &(halfedges[curr_idx]);

    prev_idx = curr_edge->prev;
    if (prev_idx == -1) return;
    prev_edge = &(halfedges[prev_idx]);

    next_idx = curr_edge->next;
    if (next_idx == -1) return;
    next_edge = &(halfedges[next_idx]);

    curr_vertex = &(vertices[(curr_edge->vertex)]);
    prev_vertex = &(vertices[(prev_edge->vertex)]);
    next_vertex = &(vertices[(next_edge->vertex)]);

    for (k = 0; k < VECTORSIZE; ++k)
        v1[k] = (curr_vertex->position)[k];

    difference((prev_vertex->position), v1, u);
    difference((next_vertex->position), v1, v);

    cross(u, v, n);
    nn = norm(n);
    curr_face->area = 0.5*nn;

    if (nn > EPS){
        for (k = 0; k < VECTORSIZE; ++k)
            (curr_face->normal)[k] = n[k]/nn;
    } else {
        for (k = 0; k < VECTORSIZE; ++k)
            (curr_face->normal)[k] = 0;
    }
}

static void _update_face_normals(int32_t *f_idxs, halfedge_t *halfedges, vertex_t *vertices, face_t *faces, signed int n_idxs)
{
    int f_idx, j;
    for (j = 0; j < n_idxs; ++j)
    {
        f_idx = f_idxs[j];
        if (f_idx == -1)
            continue;

        update_face_normal(f_idx, halfedges, vertices, faces);
    }
}

static PyObject *update_face_normals(PyObject *self, PyObject *args)
{
    PyObject *f_idxs=0, *halfedges=0, *vertices=0, *faces=0;
    int32_t j, f_idx, n_idxs;
    halfedge_t *p_halfedges;
    face_t *p_faces;
    vertex_t *p_vertices;

    if (!PyArg_ParseTuple(args, "OOOO", &f_idxs, &halfedges, &vertices, &faces)) return NULL;
    if (!PySequence_Check(f_idxs))
    {
        PyErr_Format(PyExc_RuntimeError, "expecting an sequence  eg ... (xvals, yvals) or (xvals, yvals, zvals)");
        return NULL;
    }
    if (!PyArray_Check(halfedges) || !PyArray_ISCONTIGUOUS(halfedges))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the edge data.");
        return NULL;
    }
    if (!PyArray_Check(vertices) || !PyArray_ISCONTIGUOUS(vertices)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the vertex data.");
        return NULL;
    }
    if (!PyArray_Check(faces) || !PyArray_ISCONTIGUOUS(faces)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the face data.");
        return NULL;
    } 

    n_idxs = (int32_t)PySequence_Length(f_idxs);
    p_vertices = (vertex_t*)PyArray_GETPTR1(vertices, 0);
    p_halfedges = (halfedge_t*)PyArray_GETPTR1(halfedges, 0);
    p_faces = (face_t*)PyArray_GETPTR1(faces, 0);

    for (j = 0; j < n_idxs; ++j)
    {
        f_idx = (int32_t)PyArray_DATA(PySequence_GetItem(f_idxs, (Py_ssize_t) j));
        if (f_idx == -1)
            continue;

        update_face_normal(f_idx, p_halfedges, p_vertices, p_faces);

    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *update_all_face_normals(PyObject *self, PyObject *args)
{
    PyObject *halfedges=0, *vertices=0, *faces=0;
    int j, n_idxs;
    halfedge_t *p_halfedges;
    face_t *p_faces;
    vertex_t *p_vertices;

    n_idxs = 0;

    if (!PyArg_ParseTuple(args, "iOOO", &n_idxs, &halfedges, &vertices, &faces)) return NULL;
    if (!PyArray_Check(halfedges) || !PyArray_ISCONTIGUOUS(halfedges))
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the edge data.");
        return NULL;
    }
    if (!PyArray_Check(vertices) || !PyArray_ISCONTIGUOUS(vertices)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the vertex data.");
        return NULL;
    }
    if (!PyArray_Check(faces) || !PyArray_ISCONTIGUOUS(faces)) 
    {
        PyErr_Format(PyExc_RuntimeError, "Expecting a contiguous numpy array for the face data.");
        return NULL;
    } 

    p_vertices = (vertex_t*)PyArray_GETPTR1(vertices, 0);
    p_halfedges = (halfedge_t*)PyArray_GETPTR1(halfedges, 0);
    p_faces = (face_t*)PyArray_GETPTR1(faces, 0);

    for (j = 0; j < n_idxs; ++j)
    {
        if (p_faces[j].halfedge == -1) continue;
        update_face_normal(j, p_halfedges, p_vertices, p_faces);
    }

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef triangle_mesh_utils_methods[] = {
    {"c_update_vertex_neighbors", update_vertex_neighbors, METH_VARARGS},
    {"c_update_face_normals", update_face_normals, METH_VARARGS},
    {"c_update_all_vertex_neighbors", update_all_vertex_neighbors, METH_VARARGS},
    {"c_update_all_face_normals", update_all_face_normals, METH_VARARGS},
    {NULL, NULL, 0}  /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "triangle_mesh_utils",     /* m_name */
        "C implementations of triangle_mesh operations for speed improvement",  /* m_doc */
        -1,                  /* m_size */
        triangle_mesh_utils_methods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };

PyMODINIT_FUNC PyInit_triangle_mesh_utils(void)
{
	PyObject *m;
    m = PyModule_Create(&moduledef);
    import_array()
    return m;
}
#else
PyMODINIT_FUNC inittriangle_mesh_utils(void)
{
    PyObject *m;

    m = Py_InitModule("triangle_mesh_utils", triangle_mesh_utils_methods);
    import_array()

}
#endif