#include "triangle_mesh_utils.h"

int remesh_delete_vertex(vertex_t * vertices, int32_t v_idx){
    int i;

    if (v_idx == -1) return 0;

    vertices[v_idx].halfedge = -1;

    vertices[v_idx].valence = -1;
    vertices[v_idx].component = -1;
    vertices[v_idx].locally_manifold = -1;

    for (i =0; i< 3;i++)
    {
        vertices[v_idx].position[i] = -1;
        vertices[v_idx].normal[i] = -1;
    }

    for (i =0; i< NEIGHBORSIZE;i++)
    {
        vertices[v_idx].neighbors[i] = -1;
    }


    return 1;
}

int remesh_delete_edge(halfedge_t * halfedges, int32_t e_idx){
    if (e_idx == -1) return 0;

    halfedges[e_idx].vertex = -1;

    halfedges[e_idx].twin = -1;
    halfedges[e_idx].face = -1;
    halfedges[e_idx].next = -1;
    halfedges[e_idx].prev = -1;
    halfedges[e_idx].length = -1;
    halfedges[e_idx].component = -1;
    halfedges[e_idx].locally_manifold = -1;

    return 1;
}

int remesh_delete_face(halfedge_t * halfedges, face_t * faces, int32_t e_idx){
    halfedge_t * curr_edge;
    int32_t face_idx, i;

    if (e_idx == -1) return 0;
    
    curr_edge = &halfedges[e_idx];
    if (curr_edge->vertex == -1) return 0;

    face_idx = curr_edge->face;

    faces[face_idx].halfedge = -1;

    faces[face_idx].area = -1;
    faces[face_idx].component = -1;

    for (i =0; i< 3;i++)
    {
        faces[face_idx].normal[i] = -1;
    }

    remesh_delete_edge(halfedges, curr_edge->next);
    remesh_delete_edge(halfedges, curr_edge->prev);
    remesh_delete_edge(halfedges, e_idx);

    return 1;
}

int remesh_edge_flip(halfedge_t * halfedges, vertex_t *vertices, face_t * faces, int idx, int n_halfedges, int live_update)
{
    halfedge_t * curr_edge, * twin_edge;
    int32_t twin_idx, _prev, _twin, _next, _twin_prev, _twin_next, vc, vt, new_v0, new_v1;
    
    int locally_manifold, fast_collapse_bool, i;
    int32_t * neighbours;

    float f, p, flipped_dot;
    
    // do some sanity checking on the edge to be flipped
    if (idx == -1) return 0;

    curr_edge = &(halfedges[idx]);
    if (curr_edge->locally_manifold == 0) return 0;
    
    twin_idx = curr_edge->twin;
    if (twin_idx == -1) return 0;
    twin_edge = &halfedges[twin_idx];

    //make sure both edges have valence > 3
    vc = vertices[curr_edge->vertex].valence;
    vt = vertices[twin_edge->vertex].valence;
    if ((vc < 4)||(vt <4)) return 0; 

    _prev = curr_edge->prev;
    _next = curr_edge->next;
    _twin_prev = twin_edge->prev;
    _twin_next = twin_edge->next;

    // Calculate adjustments to the halfedges we're flipping
    new_v0 = halfedges[_next].vertex;
    new_v1 = halfedges[_twin_next].vertex;

    /* If there's already an edge between these two vertices, don't flip (preserve manifoldness)
    NOTE: This is potentially a problem if we start with a high-valence mesh. In that case, swap this
    check with the more expensive commented one below.*/

    //Check for creation of multivalent edges and prevent this (manifoldness)
    locally_manifold = vertices[curr_edge->vertex].locally_manifold && vertices[twin_edge->vertex].locally_manifold && vertices[new_v0].locally_manifold && vertices[new_v1].locally_manifold;
    fast_collapse_bool = (locally_manifold && (vc < NEIGHBORSIZE) && (vt < NEIGHBORSIZE));
    if (fast_collapse_bool)
    {
        for (i =0; i < NEIGHBORSIZE; i++)
        {
            if (vertices[new_v0].neighbors[i] == -1) continue;
            if (halfedges[vertices[new_v0].neighbors[i]].vertex == new_v1) return 0;
        }
    } else
    {
        // Do a brute-force loop over all halfedges ... 
        // Hopefully we have a manifold mesh and never hit this
        for (i = 0; i < n_halfedges; i++)
        {
            if ((halfedges[i].vertex == new_v0) && (halfedges[halfedges[i].twin].vertex == new_v1)) return 0;
        }
    }

    
    //if we were not locally manifold, conservatively flag all vertices we touch as non-manifold
    if (!locally_manifold)
    {
        vertices[curr_edge->vertex].locally_manifold = 0;
        vertices[twin_edge->vertex].locally_manifold = 0;
        vertices[new_v0].locally_manifold = 0;
        vertices[new_v1].locally_manifold = 0;
    }
    
    // _next's next and prev must be adjusted
    halfedges[_next].prev = _twin_prev;
    halfedges[_next].next = twin_idx;

    // _twin_next's next and prev must be adjusted
    halfedges[_twin_next].prev = _prev;
    halfedges[_twin_next].next = idx;

    // _prev's next and prev must be updated
    halfedges[_prev].prev = idx;
    halfedges[_prev].next = _twin_next;

    // _twin_prev's next and prev must be updated
    halfedges[_twin_prev].prev = twin_idx;
    halfedges[_twin_prev].next = _next;

    // Don't even bother with checking, just make sure to update the
    // vertex_halfedges references to halfedges we know work
    vertices[curr_edge->vertex].halfedge = _next;
    vertices[twin_edge->vertex].halfedge = _twin_next;

    // Apply adjustments to the the halfedges we're flipping
    curr_edge->vertex = new_v0;
    twin_edge->vertex = new_v1;
    curr_edge->next = _prev;
    twin_edge->next = _twin_prev;
    curr_edge->prev = _twin_next;
    twin_edge->prev = _next;

    // Update pointers
    _prev = curr_edge->prev;
    _next = curr_edge->next;
    _twin_prev = twin_edge->prev;
    _twin_next = twin_edge->next;

    // Set faces
    halfedges[_next].face = curr_edge->face;
    halfedges[_prev].face = curr_edge->face;
    halfedges[_twin_next].face = twin_edge->face;
    halfedges[_twin_prev].face = twin_edge->face;

    // Finish updating vertex_halfedges references, update face references
    vertices[curr_edge->vertex].halfedge = twin_idx;
    vertices[twin_edge->vertex].halfedge = idx;
    faces[curr_edge->face].halfedge = idx;
    faces[twin_edge->face].halfedge = twin_idx;

    if (live_update)
    {
        // Update face and vertex normals
        
        update_face_normal(curr_edge->face, halfedges, vertices, faces);
        update_face_normal(twin_edge->face, halfedges, vertices, faces);
        update_single_vertex_neighbours(curr_edge->vertex, halfedges, vertices, faces);
        update_single_vertex_neighbours(twin_edge->vertex, halfedges, vertices, faces);
        update_single_vertex_neighbours(halfedges[_next].vertex, halfedges, vertices, faces);
        update_single_vertex_neighbours(halfedges[_twin_next].vertex, halfedges, vertices, faces);

        //self._clear_flags(); //FIXME - transfer to calling function once we have one.
    }

    return 1;
}

int remesh_relax(halfedge_t *halfedges, vertex_t *vertices, int n_vertices, face_t* faces, int n_faces, float l, int n_iterations)
{
    /*
    Perform n iterations of Lloyd relaxation on the mesh.

        Parameters
        ----------
            l : float
                Regularization (damping) term, used to avoid oscillations.
            n_iterations : int
                Number of iterations to apply.
    */

   int k, v, j, i;
   vertex_t * vertex_v, * vn_j;
   float w, weight_sum;
   float centroid[3], dx[3], shift[3];
   float *normal;
   
   for (k =0; k < n_iterations; k++)
   {
        for (v = 0; v < n_vertices; v++)
        {
            vertex_v = &vertices[v];
            if (vertex_v->halfedge == -1) continue;
            
            // Don't move vertices on a boundary
            if ((halfedges[vertex_v->halfedge].twin == -1) || (halfedges[halfedges[vertex_v->halfedge].prev].twin == -1)) continue;
                
            // Can't relax efficiently if valence is too high (wait for some flips first)
            if (vertex_v->valence > NEIGHBORSIZE) continue;
            
            weight_sum = 0;
            for (i = 0; i < 3; i++) centroid[i] = 0;
            
            for (j=0; j< NEIGHBORSIZE; j++)
            {
                if (vertex_v->neighbors[j] == -1) break;
                if (halfedges[vertex_v->neighbors[j]].length == 0) continue;
                
                vn_j = &vertices[halfedges[vertex_v->neighbors[j]].vertex];

                // Weight by distance to neighbors
                w = (1./halfedges[vertex_v->neighbors[j]].length);

                for (i = 0; i < 3; i++) centroid[i] += w*vn_j->position[i];
                weight_sum += w;
            }

            //divide by weights
            if (weight_sum == 0) continue;
            for (i = 0; i < 3; i++)
            {
                centroid[i] /= weight_sum;
                dx[i] = centroid[i] - vertex_v->position[i];
            }

            normal = vertex_v->normal;

            // Update vertex positions
            shift[0] = l*( (1.0 - normal[0]*normal[0])*dx[0] - normal[0]*normal[1]*dx[1] - normal[0]*normal[2]*dx[2]);
            shift[1] = l*( -normal[0]*normal[1]*dx[0] + (1.0 - normal[1]*normal[1])*dx[1] - normal[1]*normal[2]*dx[2]);
            shift[2] = l*( -normal[0]*normal[2]*dx[0] - normal[1]*normal[2]*dx[1] + (1.0 - normal[2]*normal[2])*dx[2]);
            
            for (i = 0; i < 3; i++)
            {            
                vertex_v->position[i] += shift[i];
            }
        }

        // Now we gotta recalculate the normals
        //self._update_all_face_normals(recompute=True)
        _update_all_face_normals(n_faces, halfedges, vertices, faces);
        //self._update_all_vertex_neighbours(recompute=True) //note, also updates edge lengths
        _update_all_vertex_neighbors(n_vertices, halfedges, vertices, faces);

    }
    return 1;
}

void remesh_edge_zipper(halfedge_t * halfedges, int32_t edge1, int32_t edge2)
{
    int32_t t1, t2;

    t1 = (edge1 != -1) ? halfedges[edge1].twin : -1;
    t2 = (edge2 != -1) ? halfedges[edge2].twin : -1;

    if (t1 != -1) halfedges[t1].twin = t2;   
    if (t2 != -1) halfedges[t2].twin = t1;
}

int remesh_check_neighbour_twins(halfedge_t * halfedges, vertex_t * vertices, int32_t vertex_id)
{
    int i, nn;
    for (i = 0; i < NEIGHBORSIZE; i++)
    {
        nn = vertices[vertex_id].neighbors[i];
        if ((nn != -1) && (halfedges[nn].twin == -1)) return 0;
    }

    return 1;
}

int remesh_edge_collapse(halfedge_t * halfedges, int32_t n_halfedges, vertex_t * vertices, face_t* faces, int32_t idx, int live_update)
{
    halfedge_t * curr_halfedge;
    int32_t _prev, _next, _twin, _twin_prev, _twin_next, _dead_vertex, _live_vertex;
    int32_t _prev_twin, _prev_twin_vertex, _next_prev_twin, _next_prev_twin_vertex, _twin_next_vertex, _next_twin_twin_next, _next_twin_twin_next_vertex, face0, face1, face2, face3;
    int interior, vd, vl, locally_manifold, fast_collapse_bool;
    int i, j, twin_count, shared_vertex, dead_count;
    int32_t dead_vertices[5*NEIGHBORSIZE];
    float p[3], ndot;

    // perform a huge number of sanity checks to make sure it's safe to collapse
    if (idx == -1) return 0;
    if (halfedges[idx].locally_manifold == 0) return 0;

    curr_halfedge = &halfedges[idx];
    _prev = curr_halfedge->prev;
    _next = curr_halfedge->next;
    _twin = curr_halfedge->twin;

    //check if collapsing this edge will create a free edge
    if ((halfedges[_next].twin == -1) || (halfedges[_prev].twin == -1)) return 0;

    interior = (_twin != -1);

    if (interior)
    {
        if (!remesh_check_neighbour_twins(halfedges, vertices, curr_halfedge->vertex)) return 0;
        if (!remesh_check_neighbour_twins(halfedges, vertices, halfedges[_twin].vertex)) return 0;

        //twin_halfedge = &self._chalfedges[_twin]
        _twin_prev = halfedges[_twin].prev;
        _twin_next = halfedges[_twin].next;

        if ((halfedges[_twin_prev].twin == -1) || (halfedges[_twin_next].twin == -1)) return 0; // Collapsing this edge will create another free edge

        // Make sure we create no vertices of valence <3 (manifoldness)
        if (vertices[halfedges[_next].vertex].valence < 4) return 0;
        if (vertices[halfedges[_twin_next].vertex].valence < 4) return 0;
    }

    _dead_vertex = halfedges[_prev].vertex;
    _live_vertex = curr_halfedge->vertex;

    vl = vertices[_live_vertex].valence; 
    vd = vertices[_dead_vertex].valence;
    
    if ((vl + vd - 4) < 4) return 0;
    
    locally_manifold = vertices[_live_vertex].locally_manifold && vertices[_dead_vertex].locally_manifold;

    // Check for creation of multivalent edges and prevent this (manifoldness)
    fast_collapse_bool = (locally_manifold && (vl < NEIGHBORSIZE) && (vd < NEIGHBORSIZE));
    if (fast_collapse_bool)
    {
        // Do it the fast way if we can
        twin_count = 0;
        shared_vertex = -1;
        for (i = 0; i < NEIGHBORSIZE; i++)
        {
            if (vertices[_live_vertex].neighbors[i] == -1) break;
            for (j = 0; j < NEIGHBORSIZE; j ++)
            {
                if (vertices[_dead_vertex].neighbors[j] == -1) break;

                if (halfedges[vertices[_live_vertex].neighbors[i]].vertex == halfedges[vertices[_dead_vertex].neighbors[j]].vertex)
                {
                    if (twin_count > 2) break;

                    if ((twin_count == 0) || ((twin_count > 0) && (halfedges[vertices[_dead_vertex].neighbors[j]].vertex != shared_vertex)))
                    {
                        shared_vertex = halfedges[vertices[_live_vertex].neighbors[i]].vertex;
                        twin_count += 1;
                    }
                }
            }
            if (twin_count > 2) break;
        }

        // no more than two vertices shared by the neighbors of dead and live vertex
        if (twin_count != 2) return 0;

        // assign
        for (i = 0; i < NEIGHBORSIZE; i ++)
        {
            if (vertices[_dead_vertex].neighbors[i] == -1) continue;

            halfedges[halfedges[vertices[_dead_vertex].neighbors[i]].twin].vertex = _live_vertex;
            halfedges[halfedges[vertices[_dead_vertex].neighbors[i]].prev].vertex = _live_vertex;
        }

    }
    else
    {
        // grab the set of halfedges pointing to dead_vertices
        //FIXME!!! 
        
        dead_count = 0;
        for (i = 0; i < n_halfedges; i++)
        {
            if (halfedges[i].vertex == _dead_vertex)
            {
                dead_vertices[dead_count] = i;
                dead_count += 1;
                if (dead_count > 5*NEIGHBORSIZE)
                {
                    printf("WARNING: Way too many dead vertices: {dead_count}! Politely declining to collapse.");
                    return 0;
                }
            }
        }

        // loop over all live vertices and check for twins in dead_vertices,
        // as we do in fast_collapse
        twin_count = 0;
        shared_vertex = -1;
        for (i = 0; i < n_halfedges; i++)
        {
            if (halfedges[i].twin == -1) continue;

            if (halfedges[i].vertex == _live_vertex)
                for (j = 0; j < dead_count; j ++)
                {
                    if (halfedges[dead_vertices[j]].twin == -1) continue;

                    if (halfedges[halfedges[i].twin].vertex == halfedges[halfedges[dead_vertices[j]].twin].vertex)
                    {
                        if (twin_count > 2) break;

                        if ((twin_count == 0) || ((twin_count > 0) && (halfedges[halfedges[dead_vertices[j]].twin].vertex != shared_vertex)))
                        {
                            shared_vertex = halfedges[halfedges[i].twin].vertex;
                            twin_count += 1;
                        }
                    }
                }

                if (twin_count > 2) break;
        }

        // no more than two vertices shared by the neighbors of dead and live vertex
        if (twin_count != 2) return 0;

        // assign
        for (i = 0; i < dead_count; i ++)
            halfedges[dead_vertices[i]].vertex = _live_vertex;

    }
        
    // Collapse to the midpoint of the original edge vertices
    for (i = 0; i < 3; i++)
    {
        p[i] = 0.5*(vertices[_live_vertex].position[i] + vertices[_dead_vertex].position[i]);
    }
    
    /**************************
    ### cubic interpolation
    # keep joined vertex on surface using cubic interpolation
    */

    ndot = 0;
    for (i = 0; i < 3; i++)
    {
        ndot += (vertices[_dead_vertex].normal[i]-vertices[_live_vertex].normal[i])*(vertices[_dead_vertex].position[i] - vertices[_live_vertex].position[i]);
    }

    /* correct constant for cubic interpolation would be 0.125, but this
    # inflates the mesh - 0.0625 is emperical, and slightly deflationary
    # TODO - try interpolation on a sphere instead. */

    for (i = 0; i < 3; i++)
    {
        p[i] += 0.0625*ndot*(vertices[_dead_vertex].normal[i]+vertices[_live_vertex].normal[i]);
        vertices[_live_vertex].position[i] = p[i];
    }
    
    /* px += 0.125*ndot*(n0x + n1x)
    # py += 0.125*ndot*(n0y + n1y)
    # pz += 0.125*ndot*(n0z + n1z)*/

    /*# ## Sphere projection
    # # calculate average normal
    # vsum(&self._cvertices[_live_vertex].normal0, &self._cvertices[_dead_vertex].normal0, _vn) 
    # scalar_mult(_vn, 1.0/norm(_vn))

    # # dot left normal with new normal
    # n_dot_n = dot(&self._cvertices[_live_vertex].normal0, _vn)
    
    # # dot normal with vector between the left vertex and new vertex
    # c_dot_n = (px - self._cvertices[_live_vertex].position0)*_vn[0] + (py - self._cvertices[_live_vertex].position1)*_vn[1] + (pz - self._cvertices[_live_vertex].position2)*_vn[2]

    # alpha = c_dot_n/sqrtf(2*(max(n_dot_n, 0) + 1))
    # scalar_mult(_vn, alpha)
    # px += _vn[0]
    # py += _vn[1]
    # pz += _vn[2]

    # # end sphere proj*/
    
    // update valence of vertex we keep
    vertices[_live_vertex].valence = vl + vd - 3;
    
    // delete dead vertex
    remesh_delete_vertex(vertices, (int32_t) _dead_vertex);

    // Zipper the remaining triangles
    remesh_edge_zipper(halfedges, _next, _prev);
    if (interior) remesh_edge_zipper(halfedges, _twin_next, _twin_prev);
    
    // We need some more pointers
    // TODO: make these safer
    _prev_twin = halfedges[_prev].twin;
    _prev_twin_vertex = halfedges[_prev_twin].vertex;
    _next_prev_twin = halfedges[_prev_twin].next;
    _next_prev_twin_vertex = halfedges[_next_prev_twin].vertex;
    if (interior)
    {
        _twin_next_vertex = halfedges[_twin_next].vertex;
        _next_twin_twin_next = halfedges[halfedges[_twin_next].twin].next;
        _next_twin_twin_next_vertex = halfedges[_next_twin_twin_next].vertex;
    }   
    // Make sure we have good _vertex_halfedges references
    vertices[_live_vertex].halfedge = _prev_twin;
    vertices[_prev_twin_vertex].halfedge = _next_prev_twin;

    if (interior)
    {
        vertices[_twin_next_vertex].halfedge = halfedges[_twin_next].twin;
        vertices[_next_twin_twin_next_vertex].halfedge = halfedges[_next_twin_twin_next].next;
    }

    // Grab faces to update
    face0 = halfedges[halfedges[_next].twin].face;
    face1 = halfedges[halfedges[_prev].twin].face;
    if (interior)
    {
        face2 = halfedges[halfedges[_twin_next].twin].face;
        face3 = halfedges[halfedges[_twin_prev].twin].face;
    }

    // Delete the inner triangles
    remesh_delete_face(halfedges, faces, idx);
    if (interior) remesh_delete_face(halfedges, faces, _twin);

    if (live_update)
    {
        if (interior)
        {
            // Update faces
            
            update_face_normal(face0, halfedges, vertices, faces);
            update_face_normal(face1, halfedges, vertices, faces);
            update_face_normal(face2, halfedges, vertices, faces);
            update_face_normal(face3, halfedges, vertices, faces);
            
            update_single_vertex_neighbours(_live_vertex, halfedges, vertices, faces);
            update_single_vertex_neighbours(_prev_twin_vertex, halfedges, vertices, faces);
            update_single_vertex_neighbours(_next_prev_twin_vertex, halfedges, vertices, faces);
            update_single_vertex_neighbours(_twin_next_vertex, halfedges, vertices, faces);
        }    
        else
        {
            update_face_normal(face0, halfedges, vertices, faces);
            update_face_normal(face1, halfedges, vertices, faces);
            
            update_single_vertex_neighbours(_live_vertex, halfedges, vertices, faces);
            update_single_vertex_neighbours(_prev_twin_vertex, halfedges, vertices, faces);
            update_single_vertex_neighbours(_next_prev_twin_vertex, halfedges, vertices, faces);
        }    
        //self._clear_flags(); //FIXME
    }

    
    return 1;

}

void remesh_populate_edge(halfedge_t *halfedges, int idx, int vertex, int prev, int next, int face, int twin, int locally_manifold)
{
    //# TODO - make a macro??
    halfedges[idx].vertex = vertex;
    halfedges[idx].prev = prev;
    halfedges[idx].next = next;
    halfedges[idx].face = face;
    halfedges[idx].twin = twin;
    halfedges[idx].locally_manifold = locally_manifold;
}

int remesh_split_edge(halfedge_t * halfedges, int32_t n_halfedges, vertex_t * vertices, face_t* faces, int32_t idx, int32_t * new_edges,
                      int32_t *new_vertices, int32_t * new_faces, int n_edge_idx, int n_vertex_idx, int n_face_idx, int live_update, int upsample)
{
    halfedge_t * curr_edge;
    int _prev, _next, v0, v1, _vertex_idx, i, _twin, interior, _twin_prev, _twin_next, _face_1_idx, _face_2_idx;
    int _he_0_idx, _he_1_idx, _he_2_idx, _he_3_idx, _he_4_idx, _he_5_idx;
    float ndot;
    
    if (idx == -1) return 0;
    if (halfedges[idx].locally_manifold == 0) return 0;
    
    curr_edge = &halfedges[idx];
    _prev = curr_edge->prev;
    _next = curr_edge->next;

    // Grab the new vertex position
    v0 = curr_edge->vertex;
    v1 = halfedges[_prev].vertex;

    
    // new vertex
    // ------------
    _vertex_idx = new_vertices[n_vertex_idx];
    
    for (i=0;i<3;i++)
    {
       vertices[_vertex_idx].position[i] = 0.5*(vertices[v0].position[i] + vertices[v1].position[i]);
    }
    
    
    if (!upsample)
    {
        // keep vertex on surface by using cubic interpolation
        // along the edge
        ndot = 0;

        for (i=0; i <3; i++)
        {
            ndot += (vertices[v1].normal[i] - vertices[v0].normal[i])*(vertices[v1].position[i] - vertices[v0].position[i]);
        }

        // correct constant for cubic interpolation would be 0.125 (1/8), but this
        // inflates the mesh - 0.0625 is emperical, and slightly deflationary
        // TODO - try interpolation on a sphere instead.
        
        for (i=0; i <3; i++)
        {
            vertices[_vertex_idx].position[i] += 0.0625*ndot*(vertices[v1].normal[i] + vertices[v0].normal[i]);
        }

            
        /* _vertex[0] += 0.125*ndot*(n0x + n1x)
        _vertex[1] += 0.125*ndot*(n0y + n1y)
        _vertex[2] += 0.125*ndot*(n0z + n1z)*/

        /*# ## Sphere projection
        # # calculate average normal
        # vsum(&self._cvertices[v0].normal0, &self._cvertices[v1].normal0, _vn) 
        # scalar_mult(_vn, 1.0/norm(_vn))

        # # dot left normal with new normal
        # n_dot_n = dot(&self._cvertices[v0].normal0, _vn)
        
        # # dot normal with vector between the left vertex and new vertex
        # c_dot_n = (_vertex[0] - x0x)*_vn[0] + (_vertex[1] - x0y)*_vn[1] + (_vertex[2] - x0z)*_vn[2]

        # alpha = c_dot_n/sqrtf(2*(max(n_dot_n, 0) + 1))
        # scalar_mult(_vn, alpha)
        # _vertex[0] += _vn[0]
        # _vertex[1] += _vn[1]
        # _vertex[2] += _vn[2]*/
    }

    _twin = curr_edge->twin;
    interior = (_twin != -1);  // Are we on a boundary?
    
    // Ensure the original faces have the correct pointers and add two new faces
    faces[curr_edge->face].halfedge = idx;
    
    if (interior)
    {
        _twin_prev = halfedges[_twin].prev;
        _twin_next = halfedges[_twin].next;

        faces[halfedges[_twin].face].halfedge = _twin;
        _face_1_idx = new_faces[n_face_idx++]; //self._new_face(_twin_prev)
        faces[_face_1_idx].halfedge = _twin_prev;
        halfedges[_twin_prev].face = _face_1_idx;
    }
    
    _face_2_idx = new_faces[n_face_idx++];
    //n_face_idx += 1; //self._new_face(_next)
    faces[_face_2_idx].halfedge = _next;
    halfedges[_next].face = _face_2_idx;

    // Insert the new faces
    _he_0_idx = new_edges[n_edge_idx++];
    _he_4_idx = new_edges[n_edge_idx++];
    _he_5_idx = new_edges[n_edge_idx++];
    
    if (interior)
    {
        _he_1_idx = new_edges[n_edge_idx++];
        _he_2_idx = new_edges[n_edge_idx++];
        _he_3_idx = new_edges[n_edge_idx++];

        remesh_populate_edge(halfedges, _he_1_idx, _vertex_idx, _twin_next, _twin, halfedges[_twin].face, _he_2_idx, 1);
        remesh_populate_edge(halfedges, _he_2_idx, halfedges[_twin_next].vertex, _he_3_idx, _twin_prev, _face_1_idx, _he_1_idx, 1);
        remesh_populate_edge(halfedges, _he_3_idx,_vertex_idx, _twin_prev, _he_2_idx, _face_1_idx, _he_4_idx, 1);
    } else
    {
        _he_1_idx = -1;
        _he_2_idx = -1;
        _he_3_idx = -1;
    }
    
    remesh_populate_edge(halfedges, _he_0_idx, halfedges[_next].vertex, idx, _prev, halfedges[idx].face, _he_5_idx, 1);
    remesh_populate_edge(halfedges, _he_4_idx, halfedges[idx].vertex, _he_5_idx, _next, _face_2_idx, _he_3_idx, 1);
    remesh_populate_edge(halfedges, _he_5_idx, _vertex_idx, _next, _he_4_idx, _face_2_idx, _he_0_idx, 1);

    // Update _prev, next
    halfedges[_prev].prev = _he_0_idx;
    halfedges[_next].prev = _he_4_idx;
    halfedges[_next].next = _he_5_idx;

    if (interior)
    {
        // Update _twin_next, _twin_prev
        halfedges[_twin_next].next = _he_1_idx;
        halfedges[_twin_prev].prev = _he_2_idx;
        halfedges[_twin_prev].next = _he_3_idx;

        halfedges[_twin].prev = _he_1_idx;
    }
    
    // Update _curr and _twin
    halfedges[idx].vertex = _vertex_idx;
    halfedges[idx].next = _he_0_idx;

    // Update halfedges 
    if (interior)
    {
        vertices[halfedges[_he_2_idx].vertex].halfedge = _he_1_idx;
    }
    
    vertices[halfedges[_prev].vertex].halfedge = idx;
    vertices[halfedges[_he_4_idx].vertex].halfedge = _next;
    vertices[_vertex_idx].halfedge = _he_4_idx;
    vertices[halfedges[_he_0_idx].vertex].halfedge = _he_5_idx;

    /*if (upsample) //FIXME
    {
        // Make sure these edges emanate from the new vertex stored at _vertex_idx
        if (interior)
            self._loop_subdivision_flip_edges.extend([_he_2_idx]);
        
        self._loop_subdivision_flip_edges.extend([_he_0_idx]);
        self._loop_subdivision_new_vertices.extend([_vertex_idx]);
    }*/
    
    
    if (live_update)
    {
        if (interior)
        {    
            update_face_normal(halfedges[_he_0_idx].face, halfedges, vertices, faces);
            update_face_normal(halfedges[_he_1_idx].face, halfedges, vertices, faces);
            update_face_normal(halfedges[_he_2_idx].face, halfedges, vertices, faces);
            update_face_normal(halfedges[_he_4_idx].face, halfedges, vertices, faces);
            
            //#print('vertex_neighbours')
            update_single_vertex_neighbours(halfedges[idx].vertex, halfedges, vertices, faces);
            //#print('n1')
            update_single_vertex_neighbours(halfedges[_twin].vertex, halfedges, vertices, faces);
            //#print('n2')
            update_single_vertex_neighbours(halfedges[_he_0_idx].vertex, halfedges, vertices, faces);
            //#print('n3')
            update_single_vertex_neighbours(halfedges[_he_2_idx].vertex, halfedges, vertices, faces);
            //#print('n')
            update_single_vertex_neighbours(halfedges[_he_4_idx].vertex, halfedges, vertices, faces);
            //#print('vertex_neighbours done')
        }
        else
        {
            //self._update_face_normals([self._chalfedges[_he_0_idx].face, self._chalfedges[_he_4_idx].face])
            //self._update_vertex_neighbors([self._chalfedges[_curr].vertex, self._chalfedges[_prev].vertex, self._chalfedges[_he_0_idx].vertex, self._chalfedges[_he_4_idx].vertex])
            
            update_face_normal(halfedges[_he_0_idx].face, halfedges, vertices, faces);
            update_face_normal(halfedges[_he_4_idx].face, halfedges, vertices, faces);
            
            update_single_vertex_neighbours(halfedges[idx].vertex, halfedges, vertices, faces);
            update_single_vertex_neighbours(halfedges[_prev].vertex, halfedges, vertices, faces);
            update_single_vertex_neighbours(halfedges[_he_0_idx].vertex, halfedges, vertices, faces);
            update_single_vertex_neighbours(halfedges[_he_4_idx].vertex, halfedges, vertices, faces);
        }
        
        //self._invalidate_cached_properties(); //FIXME
    }
    return 1;
}



