// #include <stdio.h>
// #include <stdlib.h>
#include "Python.h"
#include <math.h>
#include "numpy/arrayobject.h"

#include "triangle_mesh_utils.h"

#define EPS 1e-12

inline float norm(const float *pos)
{
    float n = 0;
    int i = 0;

    for (i = 0; i < VECTORSIZE; ++i)
        n += pos[i] * pos[i];
    return sqrt(n);
}

inline void cross(const float *a, const float *b, float *n)
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

inline void difference(const float *a, const float *b, float *out)
{
    int k = 0;
    for (k=0; k < VECTORSIZE; ++k)
        out[k] = a[k] - b[k];
}

inline void vsum(const float *a, const float *b, float *out)
{
    int k = 0;
    for (k=0; k < VECTORSIZE; ++k)
        out[k] = a[k] + b[k];
}

inline float dot(const float *a, const float *b)
{
    float n = 0;
    int i = 0;

    for (i = 0; i < VECTORSIZE; ++i)
        n += a[i] * b[i];
    
    return n;
}

/*inline void scalar_mult(float *a, const float b)
{
    int i=0;
    for (i = 0; i < VECTORSIZE; ++i)
        a[i] *= b;
}*/

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
    int32_t i, k, orig_idx, curr_idx, twin_idx, next_idx, tmp;
    halfedge_t *curr_edge, *twin_edge, *next_edge;
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

    i = 0;

    //zero out neighbours and normal
    for (k = 0; k < NEIGHBORSIZE; ++k)
        (curr_vertex->neighbors)[k] = -1;

    for (k = 0; k < VECTORSIZE; ++k)
        normal[k] = 0;

    // iterate twin->next, clockwise about the vertex
    while (1)
    {
        if (i < NEIGHBORSIZE)
        {
            // assign this emanating halfedge as a neighbor
            (curr_vertex->neighbors)[i] = curr_idx;

            // add the weighted normal of the face of this
            // halfedge to the vertex normal
            curr_face = &(faces[(curr_edge->face)]);
            a = curr_face->area;
            for (k = 0; k < VECTORSIZE; ++k)
                normal[k] += ((curr_face->normal)[k])*a;
        }

        // we've dealt with this emanating halfedge, increase the valence
        ++i;

        // calculate the edge length of this halfedge
        loop_vertex = &(vertices[(curr_edge->vertex)]);
        difference((curr_vertex->position), (loop_vertex->position), position);
        l = norm(position);
        curr_edge->length = l;

        // traverse to twin, if possible
        twin_idx = curr_edge->twin;
        if (twin_idx == -1)
            break;
        twin_edge = &(halfedges[twin_idx]);
        twin_edge->length = l; // the twin has the same edge length

        // traverse to next, if possible and if it is not closing
        // the one-right neighbour loop
        curr_idx = twin_edge->next;
        if ((curr_idx == -1) || (curr_idx == orig_idx))
            break;
        curr_edge = &(halfedges[curr_idx]);
    }

    // if we hit a boundary, try the reverse direction, starting from orig_idx
    if ((twin_idx == -1) && (curr_idx != -1))
    {
        // Ideally we sweep through the neighbors in a continuous fashion,
        // so we'll reverse the order of all the edges so far.
        for (k=(i-1);k>1;k--)
        {
            tmp = (curr_vertex->neighbors)[k];
            (curr_vertex->neighbors)[k] = (curr_vertex->neighbors)[k-1];
            (curr_vertex->neighbors)[k-1] = tmp;
        }

        // reset to position 0 (now position i)
        curr_idx = orig_idx;
        curr_edge = &(halfedges[curr_idx]);

        while (1)
        {
            // traverse to twin, also grab next so we can calculate the edge
            // length in the event the twin of twin is -1
            twin_idx = curr_edge->prev;
            next_idx = curr_edge->next;
            if ((twin_idx == -1) || (next_idx == -1))
                return;
            twin_edge = &(halfedges[twin_idx]);
            next_edge = &(halfedges[next_idx]);

            // technically this should be where we increase the valence, because
            // curr_edge->prev represents a new emanating halfedge, but because
            // of how the previous loop is set up, we're already at this valence

            // calculate the edge length
            loop_vertex = &(vertices[(next_edge->vertex)]);
            difference((curr_vertex->position), (loop_vertex->position), position);
            l = norm(position);
            twin_edge->length = l;

            // traverse to twin (the next emanating halfedge)
            curr_idx = twin_edge->twin;
            if ((curr_idx == -1) || (curr_idx == orig_idx))
                break;
            curr_edge = &(halfedges[curr_idx]);
            curr_edge->length = l; // the twin has the same edge length

            if (i < NEIGHBORSIZE)
            {
                // assign this emanating halfedge as a neighbor
                (curr_vertex->neighbors)[i] = curr_idx;

                // add the weighted normal of the face of this
                // halfedge to the vertex normal
                curr_face = &(faces[(curr_edge->face)]);
                a = curr_face->area;
                for (k = 0; k < VECTORSIZE; ++k)
                    normal[k] += ((curr_face->normal)[k])*a;
            }

            // we've dealt with this emanating halfedge, increase the valence
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

static int flood_fill_star_component(int32_t h_idx, int component, halfedge_t *halfedges)
{
    int32_t curr_idx, twin_idx;
    halfedge_t *curr_edge, *twin_edge;

    // slightly backward (from other functions in this file), we expect h_idx to be 
    // incident on the vertex of choice
    twin_edge = &(halfedges[h_idx]);
    if (twin_edge->component != -1) return component; // already visited
    twin_edge->component = component;
    
    // traverse until we hit another singular edge, or return to the original edge
    while (1)
    {
        //traverse
        curr_idx = twin_edge->next;
        curr_edge = &(halfedges[curr_idx]);

        // assign components
        if (curr_edge->component != -1)
        {
            printf("We are about to assign component %d to curr_edge with component %d\n", component, curr_edge->component);
        }
        curr_edge->component = component;

        // traverse here so twin_idx != h_idx if we terminate immediately
        twin_idx = curr_edge->twin;

        // we hit another singular edge, so this component is done
        if (curr_edge->locally_manifold == 0) break;

        // we hit another boundary or closed the loop, so we're also done
        if ((twin_idx == -1) || (twin_idx == h_idx)) break;
        twin_edge = &(halfedges[twin_idx]);
        if (twin_edge->locally_manifold == 0) {
            printf("Strange! Current index was locally manifold, but twin was not.\n");
            break;
        }
        if (twin_edge->component != -1)
        {
            printf("We are about to assign component %d to twin_edge with component %d\n", component, twin_edge->component);
        }
        twin_edge->component = component;
    }

    if (twin_idx != h_idx) {
        // we didn't loop all the way around
        
        twin_edge = &(halfedges[h_idx]);
        while (1) 
        {
            if (twin_edge->locally_manifold == 0) break; // don't cross a singular edge

            // traverse
            curr_idx = twin_edge->twin;
            if (curr_idx == -1) break; // a boundary
            curr_edge = &(halfedges[curr_idx]);
            if (curr_edge->locally_manifold == 0) {
                printf("2 Strange! Current index was locally manifold, but twin was not.\n");
                break; // don't cross a singular edge
            }
            // assign component
            if (curr_edge->component != -1)
            {
                printf("Going backwards, we are about to assign component %d to curr_edge with component %d\n", component, curr_edge->component);
            }
            curr_edge->component = component;
            
            twin_idx = curr_edge->prev;
            twin_edge = &(halfedges[twin_idx]);

            // assign components
            if (twin_edge->component != -1)
            {
                printf("Going backwards, we are about to assign component %d to twin_edge with component %d\n", component, twin_edge->component);
            }
            twin_edge->component = component;
        }
    }

    return (component+1);

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