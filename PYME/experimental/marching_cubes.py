import numpy as np

# MC_EDGES and MC_TRIANGLES from http://paulbourke.net/geometry/polygonise/marchingsource.cpp
#
# We assume each cube is arranged in the Euclidean 3-space as follows, where v indicates the vertex number and e
# indicates the edge number.
#
# TODO: Confusingly, the vertex numbering here is different than in PYME.LMVis.layers.octree. This is the correct vertex numbering.
#
#           z
#           ^
#           |
#          v4 ----e4----v5
#          /|           /|
#         e7|          e5|
#        /  e8        /  |
#       v7----e6----v6   e9
#       |   |        |   |
#       |  v0----e0--|- v1---> x
#      e11 /        e10 /
#       | e3         | e1
#       |/           |/
#       v3----e2-----v2
#      /
#     y
#

# This tells us which row of MC_TRIANGLES to look at based on comparison of isosurface values to grid vertices
MC_EDGES = np.array([
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
])


# Marching cubes triangles. -1 indicates no vertex, 0-11 indicate vertex on edge 0-11. The exact position of a vertex
# along edge relates to values of vertices on either end of edge.
MC_TRIANGLES = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
    [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
    [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
    [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
    [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
    [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
    [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
    [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
    [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
    [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
    [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
    [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
    [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
    [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
    [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
    [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
    [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
])

INTERPOLATION_BITMASK = [1 << n for n in range(12)]
EDGE_BITMASK = INTERPOLATION_BITMASK[0:8]

INTERPOLATION_VERTICES = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3], [1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7]])


class MarchingCubes(object):
    """
    Base class for marching cubes. Creates a surface triangulation from a set of voxels,
    represented as 8 vertices each.

    References
    ----------
        * https://www.volume-gfx.com/volume-rendering/marching-cubes/
        * http://paulbourke.net/geometry/polygonise/
    """

    _MC_MAP_MODE_MODIFIED = False #allow us to switch between this and modified marching cubes easily

    # Datatype for storing triangles created by marching cubes. Mimics STL data structure.
    # TODO: Add ('normal', '3f4') for real
    dt = np.dtype([('normal', '3f4'), ('vertex0', '3f4'), ('vertex1', '3f4'), ('vertex2', '3f4')])

    def __init__(self, isolevel=0):
        """
        Initialize the marching cubes algorithm

        Parameters
        ----------
        isolevel : int
            Threshold determining if vertex lies inside or outside the surface.
        """
        self.vertices = None  # A (-1,8,3) set of x,y,z coordinates corresponding to cubes
        self.values = None  # The corresponding vertex values (-1, 8)
        self.triangles = None  # The list of triangles
        self.isolevel = isolevel

    def cube_index(self, values):
        """
        Takes in a set of 8 vertex values (values of v0-v7) and determines if each is above or below the isolevel.

        Parameters
        ----------
        values : np.array
            Values at the 8 vertices v0-v7 corresponding to if the box is inside/outside the volume.

        Returns
        -------
        indices: int
            Value to use for lookup in MC_EDGES, MC_TRIANGLES.
        """

        # index = 0
        #
        # for val in range(len(values)):
        #     if values[val] < self.isolevel:
        #         index |= 1 << val
        #
        # return index
        return ((values < self.isolevel) * EDGE_BITMASK).sum(1)

    def interpolate_vertex(self, v0, v1, v0_value, v1_value, i=None, j0=None, j1=None):
        """
        Interpolate triangle vertex along edge formed by v0->v1.

        Parameters
        ----------
        v0, v1 :
            Vertices of edge v0->v1.
        v0_value, v1_value:
            Scalar values at vertices v0, v1. Same values as for vertex_values in edge_index().

        Returns
        -------
        Interpolated vertex of a triangle.
        """

        # if np.abs(self.isolevel - v0_value) == 0:
        #     return v0
        # if np.abs(self.isolevel - v1_value) == 0:
        #     return v1
        # if np.abs(v0_value - v1_value) == 0:
        #     return v0
        #
        # mu = (self.isolevel - v0_value) / (v1_value - v0_value)
        # p = v0 + mu * (v1 - v0)
        #
        # return p

        # Interpolate
        mu = (self.isolevel - v0_value) / (v1_value - v0_value)
        p = v0 + np.multiply(np.repeat(mu, v0.shape[2]).reshape(-1, v0.shape[1], v0.shape[2]), (v1 - v0))

        # Now take care of mu's div by 0 cases
        idxs = np.abs(self.isolevel - v0_value) == 0
        p[idxs, :] = v0[idxs, :]
        idxs = np.abs(self.isolevel - v1_value) == 0
        p[idxs, :] = v1[idxs, :]
        idxs = np.abs(v1_value - v0_value) == 0
        p[idxs, :] = v0[idxs, :]

        return p

    def create_intersection_list(self, edge, vertices, values):
        """
        Creates a vertex for each of the 12 edges based on edge_index values.

        Parameters
        ----------
        edge : int
            MC_EDGES[cube_index]
        vertices : np.array
            Cube vertices.
        values : np.array
            Cube vertex values.

        Returns
        -------
        vertices of intersection list
        """

        # intersection_vertices = np.zeros((12, 3))  # Create one set of intersection vertices, one vertex per edge
        # if edge & 1:
        #     intersection_vertices[0] = self.interpolate_vertex(vertices[0], vertices[1], values[0], values[1])
        # if edge & 2:
        #     intersection_vertices[1] = self.interpolate_vertex(vertices[1], vertices[2], values[1], values[2])
        # if edge & 4:
        #     intersection_vertices[2] = self.interpolate_vertex(vertices[2], vertices[3], values[2], values[3])
        # if edge & 8:
        #     intersection_vertices[3] = self.interpolate_vertex(vertices[3], vertices[0], values[3], values[0])
        # if edge & 16:
        #     intersection_vertices[4] = self.interpolate_vertex(vertices[4], vertices[5], values[4], values[5])
        # if edge & 32:
        #     intersection_vertices[5] = self.interpolate_vertex(vertices[5], vertices[6], values[5], values[6])
        # if edge & 64:
        #     intersection_vertices[6] = self.interpolate_vertex(vertices[6], vertices[7], values[6], values[7])
        # if edge & 128:
        #     intersection_vertices[7] = self.interpolate_vertex(vertices[7], vertices[4], values[7], values[4])
        # if edge & 256:
        #     intersection_vertices[8] = self.interpolate_vertex(vertices[0], vertices[4], values[0], values[4])
        # if edge & 512:
        #     intersection_vertices[9] = self.interpolate_vertex(vertices[1], vertices[5], values[1], values[5])
        # if edge & 1024:
        #     intersection_vertices[10] = self.interpolate_vertex(vertices[2], vertices[6], values[2], values[6])
        # if edge & 2048:
        #     intersection_vertices[11] = self.interpolate_vertex(vertices[3], vertices[7], values[3], values[7])
        #
        # return intersection_vertices

        j0 = INTERPOLATION_VERTICES[0, :]
        j1 = INTERPOLATION_VERTICES[1, :]
        v0 = vertices[:, j0]
        v1 = vertices[:, j1]
        v0_value = values[:, j0]
        v1_value = values[:, j1]

        p = self.interpolate_vertex(v0, v1, v0_value, v1_value, j0=j0, j1=j1)

        edge_indices = (np.repeat(edge, 12).reshape(edge.shape[0], 12) & INTERPOLATION_BITMASK) == 0
        p[edge_indices, :] = 0

        return p

    def create_triangles(self, intersections, triangles):
        """
        Connect the intersection list into triangles.

        Parameters
        ----------
        intersections : np.array
            Array of edge intersection vertices.
        triangles : np.array
            Array of edge indices to form triangles.

        Returns
        -------
        None
        """
        # idx = 0
        #
        # while triangles[idx] != -1:
        #     # TODO: currently adds a normal of [1, 1, 1]
        #     self.triangles.append(([0, 0, 0], intersections[triangles[idx]].tolist(),
        #                            intersections[triangles[idx+1]].tolist(), intersections[triangles[idx+2]].tolist(),
        #                            0))
        #     idx += 3
        idxs = np.array(np.where(triangles[:, 0] != -1)).reshape(-1)
        triangles_pruned = triangles[idxs].flatten()
        idxs = np.repeat(idxs, triangles.shape[1])[triangles_pruned != -1]
        triangles_pruned = triangles_pruned[triangles_pruned != -1]
        triangles_returned = intersections[idxs, triangles_pruned].reshape(int(idxs.shape[0] // 3), 3, 3)
        normals = np.cross((triangles_returned[:, 2] - triangles_returned[:, 1]),
                           (triangles_returned[:, 0] - triangles_returned[:, 1]))
                           
        triangles_stl = np.zeros(triangles_returned.shape[0], dtype=self.dt)
        triangles_stl['vertex0'] = triangles_returned[:, 0, :]
        triangles_stl['vertex1'] = triangles_returned[:, 1, :]
        triangles_stl['vertex2'] = triangles_returned[:, 2, :]
        triangles_stl['normal'] = normals

        self.triangles = triangles_stl

        return self.triangles

    def set_vertices(self, vertices, values):
        """
        Set input vertices (self.vertices) & set values (self.values) based on input vertices to use for isosurface
        extraction.

        * self.vertices is a (-1,8,3) set of x,y,z coordinates corresponding to cubes
        * self.values are the corresponding vertex values (-1, 8)

        overridden in derived classes
        """

        self.vertices = vertices
        self.values = values

    def export_triangles(self):
        """
        Return the list of triangles as a dtyped np array for use in triangular rendering.

        Returns
        -------
        triangles : np.array
            Array of triangles of type self.dt
        """

        return self.triangles

    def march(self, return_triangles=True, **kwargs):
        """
        March over the input vertices.

        Parameters
        ----------
        return_triangles : bool
            Return the list of triangles post-march or nah?

        Returns
        -------
        Optionally, an array of triangles after marching.
        """

        cube_index = self.cube_index(self.values)
        
        #we don't need to calculate anything if all vertices of a cube are above or below the threshold
        mask = ~((cube_index == 0) | (cube_index == 0xFF))
        
        edge = MC_EDGES[cube_index[mask]]
        intersections = self.create_intersection_list(edge, self.vertices[mask,:,:], self.values[mask,:])
        triangles = MC_TRIANGLES[cube_index[mask]]
        self.create_triangles(intersections, triangles)

        # for v_index in range(self.vertices.shape[0]):
        #     values = self.values[v_index, :]
        #     cube_index = self.cube_index(values)
        #
        #     edge = MC_EDGES[cube_index]
        #     if edge == 0:
        #         continue
        #
        #     vertices = self.vertices[v_index, :, :]
        #
        #     intersections = self.create_intersection_list(edge, vertices, values)
        #     triangles = MC_TRIANGLES[cube_index]
        #     self.create_triangles(intersections, triangles)

        if return_triangles:
            return self.export_triangles()
        
class RasterMarchingCubes(MarchingCubes):
    """
    Marching cubes with some optimisations data on a regular grid
    """
    
    v_offsets = np.array([[0,0,0],
                          [1,0,0],
                          [1,1,0],
                          [0,1,0],
                          [0, 0, 1],
                          [1, 0, 1],
                          [1, 1, 1],
                          [0, 1, 1]])
    
    def __init__(self, image, isolevel=0, voxelsize=(1.,1.,1.)):
        """
        Initialize the marching cubes algorithm

        Parameters
        ----------
        isolevel : int
            Threshold determining if vertex lies inside or outside the surface.
        """
        self.image = image.squeeze().astype('f')
        self.voxelsize = voxelsize
        self.triangles = None  # The list of triangles
        self.isolevel = isolevel
        
    def cube_index(self):
        """
        Takes in a set of 8 vertex values (values of v0-v7) and determines if each is above or below the isolevel.

        Parameters
        ----------
        values : np.array
            Values at the 8 vertices v0-v7 corresponding to if the box is inside/outside the volume.

        Returns
        -------
        indices: int
            Value to use for lookup in MC_EDGES, MC_TRIANGLES.
        """

        imm = self.image < self.isolevel
        
        res = (imm[:-1,:-1,:-1] << 0) + \
              (imm[1:, :-1,:-1] << 1) + \
              (imm[1:, 1:, :-1] << 2) + \
              (imm[:-1, 1:, :-1] << 3) + \
              (imm[:-1, :-1, 1:] << 4) + \
              (imm[1:, :-1, 1:] << 5) + \
              (imm[1:, 1:, 1:] << 6) + \
              (imm[:-1, 1:, 1:] << 7)
        
        return res
    
    def gen_vertices_and_vals(self, cube_mask):
        coords = np.argwhere(cube_mask)[:,None,:] + self.v_offsets[None,:,:]
        
        values = self.image[coords[:,:,0], coords[:,:,1], coords[:,:,2]]
        
        #print(coords.shape, values.shape)
        
        return coords.astype('f')*np.array(self.voxelsize)[None,None,:], values

    def march(self, return_triangles=True, **kwargs):
        """
        March over the input vertices.

        Parameters
        ----------
        return_triangles : bool
            Return the list of triangles post-march or nah?

        Returns
        -------
        Optionally, an array of triangles after marching.
        """
    
        cube_index = self.cube_index()
        
        #print(cube_index.dtype, cube_index.max(), cube_index.min())
    
        #we don't need to calculate anything if all vertices of a cube are above or below the threshold
        mask = ~((cube_index == 0) | (cube_index == 0xFF))
        
        #print(self.image.shape, cube_index.shape, mask.shape, mask.sum())
        
        cube_index = cube_index[mask].ravel()
    
        edge = MC_EDGES[cube_index]
        intersections = self.create_intersection_list(edge, *self.gen_vertices_and_vals(mask))
        triangles = MC_TRIANGLES[cube_index]
        self.create_triangles(intersections, triangles)
    
    
        if return_triangles:
            return self.export_triangles()
        
        

def generate_sphere_voxels(radius=10):
    """
    Generate a set of voxels representing a sphere to test marching cubes.

    Parameters
    ----------
    radius : int
        Radius of the 3D sphere.

    Returns
    -------
    vertices : np.array
        The vertices of the sphere voxels, shape (-1, 8, 3).
    values : np.array
        The corresponding values at the vertices coordinates, shape (-1, 8).
    """

    cube_width = 1  # voxel step size

    vertices = []
    values = []

    for z in range(2 * radius):
        for y in range(2 * radius):
            for x in range(2 * radius):
                # Default to we're outside the sphere
                v0 = v1 = v2 = v3 = v4 = v5 = v6 = v7 = 0

                vertices.append([(x, y, z), (x + cube_width, y, z),
                                 (x + cube_width, y + cube_width, z),
                                 (x, y + cube_width, z),
                                 (x, y, z + cube_width),
                                 (x + cube_width, y, z + cube_width),
                                 (x + cube_width, y + cube_width, z + cube_width),
                                 (x, y + cube_width, z + cube_width)
                                 ])

                if ((x - radius) ** 2 + (y - radius) ** 2 +
                        (z - radius) ** 2) <= radius ** 2 / 4:
                    v0 = 1
                if ((x + cube_width - radius) ** 2 + (y - radius) ** 2 +
                        (z - radius) ** 2) <= radius ** 2 / 4:
                    v1 = 1
                if ((x + cube_width - radius) ** 2 + (y + cube_width - radius) ** 2 +
                        (z - radius) ** 2) <= radius ** 2 / 4:
                    v2 = 1
                if ((x - radius) ** 2 + (y + cube_width - radius) ** 2 +
                        (z - radius) ** 2) <= radius ** 2 / 4:
                    v3 = 1
                if ((x - radius) ** 2 + (y - radius) ** 2 +
                        (z + cube_width - radius) ** 2) <= radius ** 2 / 4:
                    v4 = 1
                if ((x + cube_width - radius) ** 2 + (y - radius) ** 2 +
                        (z + cube_width - radius) ** 2) <= radius ** 2 / 4:
                    v5 = 1
                if ((x + cube_width - radius) ** 2 + (y + cube_width - radius) ** 2 +
                        (z + cube_width - radius) ** 2) <= radius ** 2 / 4:
                    v6 = 1
                if ((x - radius) ** 2 + (y + cube_width - radius) ** 2 +
                        (z + cube_width - radius) ** 2) <= radius ** 2 / 4:
                    v7 = 1

                values.append([v0, v1, v2, v3, v4, v5, v6, v7])

    return np.array(vertices), np.array(values)


def generate_sphere_image(radius=10):
    """Generate a 3D volume image of a sphere. Note that the intensity tapers at the radius (rather than ending abruptly)
    so that marching cubes can do it's thing properly"""
    X, Y, Z = np.mgrid[(-1.5*radius):(1.5*radius):1.0, (-1.5*radius):(1.5*radius):1.0, (-1.5*radius):(1.5*radius):1.0]

    # Uncomment to create sphere missing a top
    # X += radius/2
    # Y += radius/5
    # Z += radius/5

    R2 = np.sqrt(X*X + Y*Y + Z*Z)
    
    S = np.tanh(R2 - radius)
    
    return S

def image_to_vertex_values(im, voxelsize=1.0):
    """
    Translate a volume image into a series of vertices and values for marching cubes.
    
    Note: this is for debugging and uses ~32x the memory that is required to store the image. In practical applications
    it might be more efficient to generate these vertices and values inside a modified version of the marching cubes code
    
    Parameters
    ----------
    im
    voxelsize

    Returns
    -------

    """
    if np.isscalar(voxelsize):
        V = voxelsize*np.mgrid[0:im.shape[0], 0:im.shape[1], 0:im.shape[2]]
    else:
        V = np.array(voxelsize)[:,None,None,None]*np.mgrid[0:im.shape[0], 0:im.shape[1], 0:im.shape[2]]
    
    def _rs(v):
        return v.reshape(3, -1).T.reshape(-1, 1, 3)
    
    def _rs_i(i):
        return i.ravel().reshape(-1, 1)

    
    vertices = np.concatenate([_rs(V[:, :-1, :-1, :-1]),
                               _rs(V[:, 1:, :-1, :-1]),
                               _rs(V[:, 1:, 1:, :-1]),
                               _rs(V[:, :-1, 1:, :-1]),
                               _rs(V[:, :-1, :-1, 1:]),
                               _rs(V[:, 1:, :-1, 1:]),
                               _rs(V[:, 1:, 1:, 1:]),
                               _rs(V[:, :-1, 1:, 1:])], 1)

    values = np.concatenate([_rs_i(im[:-1, :-1, :-1]),
                               _rs_i(im[1:, :-1, :-1]),
                               _rs_i(im[1:, 1:, :-1]),
                               _rs_i(im[:-1, 1:, :-1]),
                               _rs_i(im[:-1, :-1, 1:]),
                               _rs_i(im[1:, :-1, 1:]),
                               _rs_i(im[1:, 1:, 1:]),
                               _rs_i(im[:-1, 1:, 1:])], 1)
    
    return vertices, values
