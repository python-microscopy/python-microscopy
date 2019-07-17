#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "triangle_mesh_utils.h"

float norm(float *pos)
{
    int i;
    float n = 0;
    for (i = 0; i < 3; ++i)
        n += pos[i] * pos[i];
    return sqrt(n);
}

void update_vertex_neighbors(signed int *v_idxs, halfedge_t *halfedges, vertex_t *vertices, face_t *faces, signed int n_idxs)
{
    signed int i, j, k, v_idx, orig_idx, curr_idx, twin_idx;
    halfedge_t *curr_edge, *twin_edge;
    vertex_t *curr_vertex;
    face_t *curr_face;

    float position[3];
    float normal[3];
    float a, l, nn;

    for (j = 0; j < n_idxs; ++j)
    {
        v_idx = v_idxs[j];
        curr_vertex = &(vertices[v_idx]);

        orig_idx = curr_vertex->halfedge;
        curr_idx = orig_idx;

        curr_edge = &(halfedges[curr_idx]);
        twin_idx = curr_edge->twin;
        twin_edge = &(halfedges[twin_idx]);

        i = 0;

        for (k = 0; k < 8; ++k)
            (curr_vertex->neighbors)[k] = -1;

        for (k = 0; k < 3; ++k)
            normal[k] = 0;

        while (1)
        {
            if ((curr_idx == -1) || (twin_idx == -1))
                break;

            if (i < 8)
            {
                (curr_vertex->neighbors)[i] = curr_idx;
                curr_face = &(faces[curr_edge->face]);
                a = curr_face->area;
                for (k = 0; k < 3; ++k) 
                    normal[k] += ((curr_face->normal)[k])*a;
            }

            for (k = 0; k < 3; ++k) 
                position[k] = (curr_vertex->position)[k] - vertices[curr_edge->vertex].position[k];

            l = norm(position);
            curr_edge->length = l;
            twin_edge->length = l;

            curr_idx = twin_edge->next;
            curr_edge = &(halfedges[curr_idx]);
            twin_idx = curr_edge->twin;
            twin_edge = &(halfedges[twin_idx]);

            ++i;

            if (curr_idx == orig_idx)
                break;
        }

        curr_vertex->valence = i;

        nn = norm(normal);
        if (nn > 0) {
            for (k = 0; k < 3; ++k)
                (curr_vertex->normal)[k] = normal[k]/nn;
        } else {
            for (k = 0; k < 3; ++k)
                (curr_vertex->normal)[k] = 0;
        }
    }
}