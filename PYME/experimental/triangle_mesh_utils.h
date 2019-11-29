#ifndef _triangle_mesh_utils_h_
#define _triangle_mesh_utils_h_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VECTORSIZE 3
// Note this must match NEIGHBORSIZE in triangle_mesh.py
// [DB - can we export this constant from the module and then use that in triangle_mesh.py so that we don't need to define it in two places?]
#define NEIGHBORSIZE 20

typedef struct {
    int32_t vertex;
    int32_t face;
    int32_t twin;
    int32_t next;
    int32_t prev;
    float length;
    int32_t component;
} halfedge_t;

typedef struct {
    int32_t halfedge;
    float normal[VECTORSIZE];
    float area;
    int32_t component;
} face_t;

typedef struct {
    float position[VECTORSIZE];
    float normal[VECTORSIZE];
    int32_t halfedge;
    int32_t valence;
    int32_t neighbors[NEIGHBORSIZE];
    int32_t component;
} vertex_t;

float norm(const float *vertex);
void cross(const float *a, const float *b, float *n);
void difference(const float *a, const float *b, float *d);

static PyObject *update_vertex_neighbors(PyObject *self, PyObject *args);
static PyObject *update_face_normals(PyObject *self, PyObject *args);

#ifdef __cplusplus
}
#endif

#endif /* _triangle_mesh_utils_h_ */