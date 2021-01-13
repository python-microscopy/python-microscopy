import numpy as np
# from PYME.experimental.modified_marching_cubes import RasterMarchingCubes

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

TRI_EDGES = np.array([
    [0,1],
    [0,2],
    [0,3],
    [1,2],
    [1,3],
    [2,3], 
])

TRI_TRIANGLES = np.array([
    [-1, -1, -1, -1, -1, -1],
    [0, 1, 2, -1, -1, -1],
    [0, 4, 3, -1, -1, -1],
    [1, 2, 4, 1, 4, 3],
    [1, 3, 5, -1, -1, -1],
    [0, 5, 2, 0, 3, 5],
    [0, 4, 5, 0, 5, 1],
    [2, 4, 5, -1, -1, -1],
    [2, 5, 4, -1, -1, -1],
    [0, 5, 4, 0, 1, 5],
    [0, 2, 5, 0, 5, 3],
    [1, 5, 3, -1, -1, -1],
    [1, 4, 2, 1, 3, 4],
    [0, 3, 4, -1, -1, -1],
    [0, 2, 1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1]
])

# TRI_TRIANGLES = np.array([
#     [-1, -1, -1, -1, -1, -1],
#     [0, 2, 1, -1, -1, -1],
#     [0, 3, 4, -1, -1, -1],
#     [1, 4, 2, 1, 3, 4],
#     [1, 5, 3, -1, -1, -1],
#     [0, 2, 5, 0, 5, 3],
#     [0, 5, 4, 0, 1, 5],
#     [2, 5, 4, -1, -1, -1],
#     [2, 4, 5, -1, -1, -1],
#     [0, 4, 5, 0, 5, 1],
#     [0, 5, 2, 0, 3, 5],
#     [1, 3, 5, -1, -1, -1],
#     [1, 2, 4, 1, 4, 3],
#     [0, 4, 3, -1, -1, -1],
#     [1, 2, 0, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1]
# ])

# TRI_TRIANGLES = np.array([
#     [-1, -1, -1, -1, -1, -1],
#     [0, 1, 2, -1, -1, -1],
#     [0, 4, 3, -1, -1, -1],
#     [2, 1, 4, 4, 3, 1],
#     [1, 3, 5, -1, -1, -1],
#     [0, 5, 2, 0, 3, 5],
#     [0, 4, 5, 0, 1, 5],
#     [2, 5, 4, -1, -1, -1],
#     [2, 5, 4, -1, -1, -1],
#     [0, 4, 5, 0, 1, 5],
#     [0, 5, 2, 0, 3, 5],
#     [1, 3, 5, -1, -1, -1],
#     [2, 1, 4, 4, 3, 1],
#     [0, 4, 3, -1, -1, -1],
#     [0, 1, 2, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1]
# ])

# TRI_EDGES = np.array([
#     [0,0], # 0
#     [0,1], # 1
#     [0,2], # 2
#     [0,3], # 3
#     [1,0], # 4
#     [1,1], # 5
#     [1,2], # 6
#     [1,3], # 7
#     [2,0], # 8
#     [2,1], # 9
#     [2,2], # 10
#     [2,3], # 11
#     [3,0], # 12
#     [3,1], # 13
#     [3,2], # 14
#     [3,3]  # 15
# ])

# TRI_TRIANGLES = np.array([
#     [-1, -1, -1, -1, -1, -1],
#     [1, 2, 3, -1, -1, -1],
#     [4, 7, 6, -1, -1, -1],
#     [3, 2, 7, 7, 6, 2],
#     [8, 9, 11, -1, -1, -1],
#     [1, 11, 3, 1, 6, 11],
#     [1, 7, 11, 1, 2, 11],
#     [12, 14, 13, -1, -1, -1],
#     [12, 14, 13, -1, -1, -1],
#     [1, 7, 11, 1, 2, 11],
#     [1, 11, 3, 1, 6, 11],
#     [8, 9, 11, -1, -1, -1],
#     [3, 2, 7, 7, 6, 2],
#     [4, 7, 6, -1, -1, -1],
#     [1, 2, 3, -1, -1, -1],
#     [-1, -1, -1, -1, -1, -1]
# ])

class MarchingTetrahedra(object):
    """
    Creates a surface triangulation from a set of tetrahedra.

    References
    ----------
        1. http://paulbourke.net/geometry/polygonise/
        2. Jules Bloomenthal, An Implicit Surface Polygonizer, 
           Graphics Gems IV, Paul Heckbert (editor), pages 324-349, 
           Academic Press (Boston, Massachusetts), 1994
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
        # Consistently sort
        switch = v0_value > v1_value

        # Interpolate along the edge v0 -> v1
        mu0 = 1.*(self._isolevel - v0_value) / (v1_value - v0_value)
        mu1 = 1.*(self._isolevel - v1_value) / (v0_value - v1_value)
        p0 = v0 + mu0[..., None]*(v1-v0)
        p1 = v1 + mu1[..., None]*(v0-v1)

        p = switch[...,None]*p0 + (1-switch[...,None])*p1

        # Are v0 and v1 the same vertex?
        # If so, choose v0 as the triangle vertex position.
        eps = 1e-6
        idxs = (np.abs(v1_value - v0_value) < eps) | \
                (np.abs(v1_value - self._isolevel) < eps) | \
                (np.abs(v0_value - self._isolevel) < eps)
        p[idxs, :] = v0[idxs, :]

        # p = v0

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


# Below is a single voxel
#     z
#     ^
#     |
#    v4 ----------v6
#    /|           /|
#   / |          / |
#  v5----------v7  |
#  |  |    c    |  |
#  | v0---------|-v2
#  | /          | /
#  v1-----------v3---> y
#  /
# x
#
# Now note that the oriented tetrahedra are
#
# v0 v4 v6 v7
# v0 v5 v4 v7
# v0 v6 v2 v7
# v0 v2 v3 v7
# v0 v1 v5 v7
# v0 v3 v1 v7

# Set the offsets moving along the 3D grid from index 1 to -1
# on each axis
V_OFFSETS = np.array([[0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1]])

# Split each voxel into tetrahedron

# VOX_TETS = np.array([
#     [0,4,6,7],
#     [0,5,4,7],
#     [0,6,2,7],
#     [0,2,3,7],
#     [0,1,5,7],
#     [0,3,1,7]
# ])

# VOX_TETS = np.array([
#     [0,3,2,6],
#     [0,3,7,6],
#     [0,4,7,6],
#     [0,7,1,3],
#     [0,7,1,4],
#     [5,7,1,4]
# ])

VOX_TETS = np.array([
    [0,3,6,2],
    [0,3,7,6],
    [0,4,6,7],
    [0,7,3,1],
    [0,7,1,4],
    [5,7,4,1]
])

class RasterMarchingTetrahedra(MarchingTetrahedra):
    def __init__(self, image, isolevel=0, voxelsize=(1., 1., 1.)):
        """Marching tetrahedra on a regular grid. Splits the grid
        into tetrahedron and then marches.

        TODO: This is purposefully missing some obvious optimizations
        for testing purposes. I wanted to check that the algorithm
        works before putting optimizations in.

        Parameters
        ----------
        image : np.array
            3D array of regularly spaced voxels.
        isolevel : int, optional
            Threshold determining if vertex lies inside or outside the surface, by default 0
        voxelsize : tuple, optional
            Voxelsize of the image, by default (1., 1., 1.)
        """

        self.image = image.squeeze().astype('f')
        self.voxelsize = voxelsize
        MarchingTetrahedra.__init__(self, None, None, isolevel)
        
    def gen_vertices_and_vals(self):
        xx, yy, zz = np.meshgrid(np.arange(self.image.shape[0]-1), 
                                 np.arange(self.image.shape[1]-1), 
                                 np.arange(self.image.shape[2]-1))

        coords = np.vstack([xx.ravel(),yy.ravel(),zz.ravel()]).T[:, None, :] + V_OFFSETS[None, :, :]

        vertices = coords.astype('f') * np.array(self.voxelsize)[None, None, :]
        
        values = self.image[coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]]
        
        # print(coords.shape, values.shape)

        vertices = vertices[:,VOX_TETS,:].reshape(-1,4,3,order='C')
        values = values[:,VOX_TETS].reshape(-1,4,order='C')
        
        return vertices, values

    def march(self, return_triangles=True):        
        self._vertices, self._values = self.gen_vertices_and_vals()
        return MarchingTetrahedra.march(self, return_triangles)

# class RasterMarchingTetrahedraCubes(RasterMarchingCubes):
#     def gen_vertices_and_vals(self, cube_mask):

#         # Transform from voxels to tetrahedron
#         xx, yy, zz = np.meshgrid(np.arange(self.image.shape[0]-1), 
#                                  np.arange(self.image.shape[1]-1), 
#                                  np.arange(self.image.shape[2]-1))

#         coords = np.vstack([xx.ravel(),yy.ravel(),zz.ravel()]).T[:, None, :] + V_OFFSETS[None, :, :]

#         vertices = coords.astype('f') * np.array(self.voxelsize)[None, None, :]
        
#         values = self.image[coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]]
        
#         # print(coords.shape, values.shape)

#         vertices = vertices[:,VOX_TETS,:].reshape(-1,4,3,order='C')
#         values = values[:,VOX_TETS].reshape(-1,4,order='C')

#         # Now transform back into degenerate voxels
#         TET_TO_VOX = np.array([0, 0, 1, 1, 2, 3, 2, 3])

#         vertices = vertices[:,TET_TO_VOX]
#         values = values[:,TET_TO_VOX]
        
#         return vertices, values
        
