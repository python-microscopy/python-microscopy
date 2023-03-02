#include "triangle_mesh_utils.h"
int remesh_edge_flip(halfedge_t * halfedges, vertex_t *vertices, face_t * faces, int idx, int n_halfedges, int live_update);
int remesh_relax(halfedge_t *halfedges, vertex_t *vertices, int n_vertices, face_t* faces, int n_faces, float l, int n_iterations);
void remesh_edge_zipper(halfedge_t * halfedges, int32_t edge1, int32_t edge2);
int remesh_check_neighour_twins(halfedge_t * halfedges, vertex_t * vertices, int32_t vertex_id);
int remesh_edge_collapse(halfedge_t * halfedges, int32_t n_halfedges, vertex_t * vertices, face_t* faces, int32_t idx, int live_update);
int remesh_delete_vertex(vertex_t * vertices, int32_t v_idx);
int remesh_delete_edge(halfedge_t * halfedges, int32_t e_idx);
int remesh_delete_face(halfedge_t * halfedges, face_t * faces, int32_t e_idx);