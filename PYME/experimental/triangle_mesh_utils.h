#ifndef _triangle_mesh_utils_h_

#define _triangle_mesh_utils_h_

#ifdef __cplusplus
extern "C" {
#endif

#define VECTORSIZE 3
// Note this must match NEIGHBORSIZE in triangle_mesh.py
#define NEIGHBORSIZE 20

typedef struct {
    signed int vertex;
    signed int face;
    signed int twin;
    signed int next;
    signed int prev;
    float length;
} halfedge_t;

typedef struct {
    signed int halfedge;
    float normal[VECTORSIZE];
    float area;
} face_t;

typedef struct {
    float position[VECTORSIZE];
    float normal[VECTORSIZE];
    signed int halfedge;
    signed int valence;
    signed int neighbors[NEIGHBORSIZE];
} vertex_t;

float norm(float *vertex);
void cross(float *a, float *b, float *n);
// void update_vertex_neighbors(signed int *v_idxs, halfedge_t *halfedges, vertex_t *vertices, face_t *faces, signed int n_idxs);
// void update_face_normals(signed int *f_idxs, halfedge_t *halfedges, vertex_t *vertices, face_t *faces, signed int n_idxs);
static PyObject *update_vertex_neighbors(PyObject *self, PyObject *args);
static PyObject *update_face_normals(PyObject *self, PyObject *args);

#ifdef __cplusplus
}
#endif

#endif /* _triangle_mesh_utils_h_ */