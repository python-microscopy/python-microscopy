import numpy as np

#           + 0
#          /|\
#         / | \
#        /  |  \
#      e2   |   e0
#      /   e1    \
#     /     |     \
#  3 +--e4-----e4--+ 1
#     \     |     /
#      \   e1    /
#      e5   |   e3
#        \  |  /
#         \ | /
#          \|/
#           + 2


INTERPOLATION_BITMASK = [1 << n for n in range(12)]
TRI_BITMASK = INTERPOLATION_BITMASK[0:4]

# TRI_CLASS = np.array([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F])

TRI_EDGES = np.array([
    [0,1],
    [0,2],
    [0,3],
    [1,2],
    [1,3],
    [2,3]
])

TRI_TRIANGLES = np.array([
    [-1, -1, -1, -1, -1, -1],
    [0, 1, 2, -1, -1, -1],
    [0, 4, 3, -1, -1, -1],
    [2, 1, 4, 4, 3, 1],
    [1, 3, 5, -1, -1, -1],
    [0, 5, 2, 0, 3, 5],
    [0, 4, 5, 0, 1, 5],
    [2, 5, 4, -1, -1, -1],
    [2, 5, 4, -1, -1, -1],
    [0, 4, 5, 0, 1, 5],
    [0, 5, 2, 0, 3, 5],
    [1, 3, 5, -1, -1, -1],
    [2, 1, 4, 4, 3, 1],
    [0, 4, 3, -1, -1, -1],
    [0, 1, 2, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1]
])

class MarchingTetrahedra(object):
    """
    Creates a surface triangulation from a set of tetrahedra.

    References
    ----------
        1. http://paulbourke.net/geometry/polygonise/
    """

    def __init__(self, vertices=None, values=None, isolevel=0):

        self._vertices = vertices # A (-1,4,3) set of x,y,z coordinates corresponding to cubes
        self._values = values # The corresponding vertex values (-1, 4)
        self._triangles = None
        self._isolevel = isolevel

        self.dt = np.dtype([('normal', '3f4'), ('vertex0', '3f4'), ('vertex1', '3f4'), 
                            ('vertex2', '3f4'), ('attrib', 'u2')])

    def tri_index(self, values):
        return ((values < self._isolevel) * TRI_BITMASK).sum(1)

    def interpolate_vertex(self, v0, v1, v0_value, v1_value, i=None, j0=None, j1=None):
        # Interpolate along the edge v0 -> v1
        mu = 1.*(self._isolevel - v0_value) / (v1_value - v0_value)
        p = v0 + mu[..., None]*(v1-v0)

        # Are v0 and v1 the same vertex?
        # If so, choose v0 as the triangle vertex position.
        idxs = (np.abs(v1_value - v0_value) < 1e-12)
        p[idxs, :] = v0[idxs, :]

        return p

    def march(self, return_triangles=True):
        tri_idxs = self.tri_index(self._values)
        edge_idxs = TRI_TRIANGLES[tri_idxs]
        edges_mask = (edge_idxs != -1)
        _, xx = np.meshgrid(np.arange(6),np.arange(len(self._vertices)))

        j0 = TRI_EDGES[:,0]
        j1 = TRI_EDGES[:,1]
        
        v0 = self._vertices[:,j0]
        v1 = self._vertices[:,j1]
        v0_value = self._values[:,j0]
        v1_value = self._values[:,j1]

        p = self.interpolate_vertex(v0, v1, v0_value, v1_value, j0=j0, j1=j1)
        
        triangles = p[xx, edge_idxs][edges_mask].reshape(-1,3,3)
        normals = np.cross((triangles[:, 2] - triangles[:, 1]),
                           (triangles[:, 0] - triangles[:, 1]))
        
        triangles_stl = np.zeros(triangles.shape[0], dtype=self.dt)
        triangles_stl['vertex0'] = triangles[:, 0, :]
        triangles_stl['vertex1'] = triangles[:, 1, :]
        triangles_stl['vertex2'] = triangles[:, 2, :]
        triangles_stl['normal'] = normals

        self._triangles = triangles_stl
        
        if return_triangles:
            return self._triangles