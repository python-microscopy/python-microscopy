import numpy as np


def stl_to_verts_inds(triangles_stl):
    """
    Convert a set of STL mesh triangles to a set of ordered vertices and
    indices. This allows us to treat triangles as connected.

    Parameters
    ----------
    triangles_stl : np.array
        Array of dtype np.dtype([('normal', '3f4'), ('vertex0', '3f4'), ('vertex1', '3f4'), ('vertex2', '3f4')])

    Returns
    -------
    vertices : np.array
        Vector of unique vertices of triangles in mesh.
    normals : np.array
        Vector of normals associated with vertices.
    indices : np.array
        Ordered vector of triangle indices
    """

    vertices_raw = np.vstack((triangles_stl['vertex0'], triangles_stl['vertex1'], triangles_stl['vertex2']))
    vertices, indices_raw = np.unique(vertices_raw, return_inverse=True, axis=0)
    indices = indices_raw.reshape(int(indices_raw.shape[0] / 3), 3, order='F')
    normals = triangles_stl['normal']

    return vertices, normals, indices


def verts_inds_to_stl(vertices, normals, indices):
    """
    Convert vertices, indices representation back to STL.
    """

    dt = np.dtype([('normal', '3f4'), ('vertex0', '3f4'), ('vertex1', '3f4'), ('vertex2', '3f4')])

    triangles_stl = np.zeros(normals.shape[0], dtype=dt)
    triangles_stl['vertex0'] = vertices[indices[:, 0]]
    triangles_stl['vertex1'] = vertices[indices[:, 1]]
    triangles_stl['vertex2'] = vertices[indices[:, 2]]
    triangles_stl['normal'] = normals

    return triangles_stl


def smooth_normals(normals, indices):
    """
    Takes an organized mesh (output of stl_to_verts_inds) and smooths the normals
    by interpolating over neighboring vertices.

    Parameters
    ----------
    normals, indices : np.array
        Output of stl_to_verts_inds.

    Returns
    -------
    normals : np.array
        Smoothed normals, interpolating from neighboring normals.
    """

    # General idea:
    # - Grab the index of the current vertex
    # - Find all triangles containing this index
    # - Grab the normals from all vertices in the resulting triangles
    # - Average these normals, replace the current normal with the average
    #   (not the cleverest way of doing this)
    # - Note that by "replace" the current normal, we mean do so in a
    #   duplicate of the original normal array, which we must keep intact so we
    #   don't get different results from changing the order of the smoothing

    new_normals = np.copy(normals)  # Generate a copy of normals
    # TODO: This isn't vectorized
    for idx in range(indices.shape[0]):
        tri = indices[idx]
        new_normals[idx] = np.mean(np.vstack((normals[
                                              np.where(indices == tri[0])[0],
                                              :], normals[
                                                  np.where(indices == tri[1])[
                                                      0], :], normals[np.where(
            indices == tri[2])[0], :])), axis=0)

    return new_normals

# Need edge_collapse, edge_swap, add_edge, add_vertex