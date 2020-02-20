import pytest
pytestmark = pytest.mark.skip(reason="segfaults on linux (needs modification for updated code)")

import numpy as np

from PYME.experimental import marching_cubes

# from PYME.experimental import triangle_mesh
from PYME.experimental import _triangle_mesh as triangle_mesh

# Ground truth faces and halfedges for a K3 planar graph
K3_FACES = np.array([[2,0,1]])
K3_H_VERTEX = np.array([0,1,2])
K3_H_FACE = np.array([0,0,0])
K3_H_TWIN = np.array([-1,-1,-1])
K3_H_NEXT = np.array([1,2,0])
K3_H_PREV = np.array([2,0,1])
K3_NEIGHBORS = np.array([[1,2],[0,2],[0,1]])

# Ground truth faces and halfedges for a K4 collapsed planar graph
# (K3 plus some extra edges)
K4_COLLAPSED_FACES = np.array([[2,0,1],[1,0,2]])
K4_COLLAPSED_H_VERTEX = np.array([0,0,1,1,2,2])
K4_COLLAPSED_H_FACE = np.array([0,3,0,3,0,3])
K4_COLLAPSED_H_TWIN = np.array([5,2,1,4,3,0])
K4_COLLAPSED_H_NEXT = np.array([2,5,4,1,0,3])
K4_COLLAPSED_H_PREV = np.array([4,3,0,5,2,1])
K4_COLLAPSED_NEIGHBORS = np.array([[1,2],[0,2],[0,1]])

# Ground truth faces and halfedges for a K4 planar graph (tetrahedron)
K4_FACES = np.array([[3,0,1],[2,0,3],[3,1,2],[1,0,2]])
K4_H_VERTEX = np.array([0,0,0,1,1,1,2,2,2,3,3,3])
K4_H_FACE = np.array([0,1,3,0,2,3,1,2,3,0,1,2])
K4_H_TWIN = np.array([10,8,3,2,9,7,11,5,1,4,0,6])
K4_H_NEXT = np.array([3,10,8,9,7,2,1,11,5,0,6,4])
K4_H_PREV = np.array([9,6,5,0,11,8,10,4,2,3,1,7])
K4_NEIGHBORS = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])

# Ground truth for edge pre- and post-flip
PRE_FLIP_FACES = np.array([[3,0,1],[3,1,2]])
PRE_FLIP_H_VERTEX = np.array([0,1,1,2,3,3])
PRE_FLIP_H_FACE = np.array([0,0,1,1,0,1])
PRE_FLIP_H_TWIN = np.array([-1,-1,4,-1,2,-1])
PRE_FLIP_H_NEXT = np.array([1,4,3,5,0,2])
PRE_FLIP_H_PREV = np.array([4,0,5,2,1,3])
PRE_FLIP_NEIGHBORS = np.array([[1,3],[0,2,3],[1,3],[0,1,2]])

POST_FLIP_FACES = np.array([[2,0,1],[0,2,3]])
POST_FLIP_H_VERTEX = np.array([0,0,1,2,2,3])
POST_FLIP_H_FACE = np.array([0,1,0,0,1,1])
POST_FLIP_H_TWIN = np.array([4,-1,-1,-1,0,-1])
POST_FLIP_H_NEXT = np.array([2,4,3,0,5,1])
POST_FLIP_H_PREV = np.array([3,5,0,2,1,4])
POST_FLIP_NEIGHBORS = np.array([[1,2,3],[0,2],[0,1,3],[0,2]])

# Ground truth for pre- and post-split
PRE_SPLIT_FACES = np.array([[3,0,1],[3,1,2]])
PRE_SPLIT_H_VERTEX = np.array([0,1,1,2,3,3])
PRE_SPLIT_H_FACE = np.array([0,0,1,1,0,1])
PRE_SPLIT_H_TWIN = np.array([-1,-1,4,-1,2,-1])
PRE_SPLIT_H_NEXT = np.array([1,4,3,5,0,2])
PRE_SPLIT_H_PREV = np.array([4,0,5,2,1,3])
PRE_SPLIT_NEIGHBORS = np.array([[1,3],[0,2,3],[1,3],[0,1,2]])

POST_SPLIT_FACES = np.array([[4,0,1],[4,1,2],[4,2,3],[3,0,4]])
POST_SPLIT_H_VERTEX = np.array([0,0,1,1,2,2,3,3,4,4,4,4])
POST_SPLIT_H_FACE = np.array([0,3,0,1,1,2,2,3,0,1,2,3])
POST_SPLIT_H_TWIN = np.array([11,-1,-1,8,-1,9,-1,10,3,5,7,0])
POST_SPLIT_H_NEXT = np.array([2,11,8,4,9,6,10,1,0,3,5,7])
POST_SPLIT_H_PREV = np.array([8,7,0,9,3,10,5,11,2,4,6,1])
POST_SPLIT_NEIGHBORS = np.array([[1,3,4],[0,2,4],[1,3,4],[0,2,4],[0,1,2,3]])

PRE_SNAP_FACES = np.array([[2,0,1],[1,0,3],[2,1,4],[5,0,2],[8,6,7],[7,6,9],[8,7,10],[11,6,8]])
                            # 0,1, 2,3, 4, 5,6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
PRE_SNAP_H_VERTEX = np.array([0,0, 0,1, 1, 1,2, 2, 2, 3, 4, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9,10,11])
PRE_SNAP_H_FACE =   np.array([0,1, 3,0, 1, 2,0, 2, 3, 1, 2, 3, 4, 5, 7, 4, 5, 6, 4, 6, 7, 5, 6, 7])
PRE_SNAP_H_TWIN =   np.array([8,3,-1,1,-1, 6,5,-1, 0,-1,-1,-1,20,15,-1,13,-1,18,17,-1,12,-1,-1,-1])
PRE_SNAP_H_NEXT =   np.array([3,9, 8,6, 1,10,0, 5,11, 4, 7, 2,15,21,20,18,13,22,12,17,23,16,19,14])
PRE_SNAP_H_PREV =   np.array([6,4,11,0, 9, 7,3,10, 2, 1, 5, 8,18,16,23,12,21,19,15,22,14,13,17,20])
PRE_SNAP_H_NEIGHBORS = np.array([[1,2,3,5],[0,3,2,4],[0,1,4,5],[0,1],[1,2],[0,2],[7,8,9,11],[6,8,9,10],[6,7,10,11],[6,7],[7,8],[6,8]])

POST_SNAP_FACES = np.array([[1,0,3],[2,1,4],[5,0,2],[7,6,9],[8,7,10],[11,6,8]])
                            #  0,  1,  2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17
POST_SNAP_H_VERTEX = np.array([0,  0,  1, 1, 2, 2, 3, 4, 5, 0, 0, 2, 2, 1, 1, 9,10,11])
POST_SNAP_H_FACE =   np.array([1,  3,  1, 2, 2, 3, 1, 2, 3, 5, 7, 5, 6, 6, 7, 5, 6, 7])
POST_SNAP_H_TWIN =   np.array([14,-1, -1,12,-1, 9,-1,-1,-1, 5,-1,-1, 3,-1, 0,-1,-1,-1])
POST_SNAP_H_NEXT =   np.array([6,  5,  0, 7, 3, 8, 2, 4, 1,15,14, 9,16,12,17,11,13,10])
POST_SNAP_H_PREV =   np.array([2,  8,  6, 4, 7, 1, 0, 3, 5,11,17,15,13,16,10, 9,12,14])
POST_SNAP_H_NEIGHBORS = np.array([[1,2,3,5],[ 0,3,2,4],[0,1,4,5],[0,1],[1,2],[0,2],[7,8,9,11],[6,8,9,10],[6,7,10,11],[6,7],[7,8],[6,8]])

def _generate_vertices(num=4):
    """
    Generate a set of num random vertices (for use in
    fixed topology).
    """

    return np.random.rand(num, 3)

def _generate_test_sphere():
    """
    Create a sphere for testing purposes.
    """
    # Creates a sample sphere
    S = marching_cubes.generate_sphere_image()
    # Converts the sphere to vertices, indices (the equivalent for an image is position (x,y,z) and intensity)
    v, i = marching_cubes.image_to_vertex_values(S, voxelsize=1.)

    v -= 0.5*np.array(S.shape)[None, None, :]

    threshold = 0.5
    mc = marching_cubes.MarchingCubes(threshold)
    mc.vertices = v
    mc.values = i
    tris = mc.march()

    return tris

def _test_topology(mesh, _vertex, _face, _twin, _next, _prev):
    # Construct mapping between input vectors and algorithm ordering
    vm = (_vertex[:,None] == mesh._halfedges['vertex'][None, :])
    fm = (_face[:,None] == mesh._halfedges['face'][None, :])
    m = np.argwhere(vm & fm)[:, 1]

    t_bool = (mesh._halfedges['twin'][m] == -1) & (_twin == -1)

    twin_vert_eq = (mesh._halfedges['vertex'][mesh._halfedges['twin']][m] == _vertex[_twin]) | t_bool
    twin_face_eq = (mesh._halfedges['face'][mesh._halfedges['twin']][m] == _face[_twin]) | t_bool
    next_vert_eq = (mesh._halfedges['vertex'][mesh._halfedges['next']][m] == _vertex[_next])
    next_face_eq = (mesh._halfedges['face'][mesh._halfedges['next']][m] == _face[_next])
    prev_vert_eq = (mesh._halfedges['vertex'][mesh._halfedges['prev']][m] == _vertex[_prev])
    prev_face_eq = (mesh._halfedges['face'][mesh._halfedges['prev']][m] == _face[_prev])

    return (np.all(twin_vert_eq & twin_face_eq & next_vert_eq & next_face_eq & prev_vert_eq & prev_face_eq) & _test_halfedges(mesh))

def _test_halfedges(mesh):
    # Check that the next next vertex is the same as the prev vertex
    next_next_prev = (mesh._halfedges['vertex'][mesh._halfedges['next'][mesh._halfedges['next'][mesh._vertices['halfedge']]]] == mesh._halfedges['vertex'][mesh._halfedges['prev'][mesh._vertices['halfedge']]])

    # Check that the prev vertex is the same as the twin vertex
    twins = mesh._halfedges['twin'][mesh._vertices['halfedge']]
    t_idx = (twins != -1)
    twin_prev = (mesh._halfedges['vertex'][twins[t_idx]] == mesh._halfedges['vertex'][(mesh._halfedges['prev'][mesh._vertices['halfedge']])[t_idx]])
    
    return (np.all(next_next_prev) & np.all(twin_prev))

def _test_normals(mesh):
    """
    Check if local update operations match global update of normals.
    """

    # Get values as updated by the flip
    face_normals = np.copy(mesh.face_normals)
    vertex_normals = np.copy(mesh.vertex_normals)
    vertex_neighbors = np.copy(mesh.vertex_neighbors)

    face_mask = np.all(face_normals != -1, axis=1)
    vertex_mask = np.all(vertex_normals != -1, axis=1)
    
    # Zero out the normals/neighbors entries
    mesh._vertices['neighbors'][:] = -1
    mesh._vertices['normal'][:] = -1
    mesh._faces['normal'][:] = -1

    # Recalculate as if this is an initial calculation (this assumes 
    # mesh.face_normals and mesh.vertex_normals works, which is tested separately)
    return np.all(face_normals[face_mask] == mesh.face_normals[face_mask]) & np.all(vertex_normals[vertex_mask] == mesh.vertex_normals[vertex_mask]) & np.all(vertex_neighbors[vertex_mask] == mesh.vertex_neighbors[vertex_mask])

def test_face_normal_magnitude():
    tris = _generate_test_sphere()
    mesh = triangle_mesh.TriangleMesh.from_np_stl(tris)
    fn = mesh.face_normals
    fnn = np.linalg.norm(fn, axis=1)
    y = np.ones(fnn.shape)

    np.testing.assert_array_almost_equal(fnn, y)

def test_vertex_normal_magnitude():
    tris = _generate_test_sphere()
    mesh = triangle_mesh.TriangleMesh.from_np_stl(tris)
    vn = mesh.vertex_normals
    vnn = np.linalg.norm(vn, axis=1)
    y = np.ones(vnn.shape)

    np.testing.assert_array_almost_equal(vnn, y)

def test_vertex_normal_sign():
    tris = _generate_test_sphere()
    mesh = triangle_mesh.TriangleMesh.from_np_stl(tris)

    # np.testing.assert_array_almost_equal(np.sign(mesh.vertex_normals), 
    #                                      np.sign(mesh.vertices))

    # Due to rounding near zero, assert_array_almost_equal does not work.
    # There's usually about a 3% mismatch, which is acceptable.
    vn = mesh.vertex_normals
    assert ((np.sum(np.sign(vn)!=np.sign(mesh.vertices))/vn.size) < 0.05)

def test_get_halfedges():

    k4_vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(k4_vertices, K4_FACES)

    assert _test_topology(mesh, K4_H_VERTEX, K4_H_FACE, K4_H_TWIN, K4_H_NEXT, K4_H_PREV)

def test_vertex_neighbors():

    k4_vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(k4_vertices, K4_FACES)
    mesh._vertices['valence'][:] = 12  # cheat our manifold checks

    assert np.all(np.sort(mesh._halfedges['vertex'][mesh.vertex_neighbors[:,:K4_NEIGHBORS.shape[1]]], axis=1) == np.sort(K4_NEIGHBORS, axis=1))

def test_euler_characteristic():

    k4_vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(k4_vertices, K4_FACES)

    assert ((len(mesh.vertices) - len(mesh._halfedges['vertex'])/2. + len(mesh._faces)) == 2)

def test_missing_halfedges():

    k3_vertices = _generate_vertices(3)
    mesh = triangle_mesh.TriangleMesh(k3_vertices, K3_FACES)

    assert _test_topology(mesh, K3_H_VERTEX, K3_H_FACE, K3_H_TWIN, K3_H_NEXT, K3_H_PREV)

def test_edge_flip_topology():

    # vertices = _generate_vertices(4)
    # There's a concavity check in edge_flip, so we just stick with fixed, convex vertices
    vertices = np.array([[0.8233384 , 0.04200047, 0.9175104 ],
                         [0.12197538, 0.3638311 , 0.20577249],
                         [0.63744223, 0.55602515, 0.61852   ],
                         [0.8490536 , 0.721189  , 0.2788919 ]])
    mesh = triangle_mesh.TriangleMesh(vertices, PRE_FLIP_FACES)
    mesh._vertices['valence'][:] = 12  # cheat our manifold checks

    flip_idx = np.where((mesh._halfedges['vertex'] == 3) & (mesh._halfedges['face'] == 0))[0][0]

    mesh.edge_flip(flip_idx)

    assert(_test_topology(mesh, POST_FLIP_H_VERTEX, POST_FLIP_H_FACE, POST_FLIP_H_TWIN, POST_FLIP_H_NEXT, POST_FLIP_H_PREV))

# def test_edge_flip_normals():

#     vertices = _generate_vertices(4)
#     mesh = triangle_mesh.TriangleMesh(vertices, PRE_FLIP_FACES)
#     mesh._vertices['valence'][:] = 12  # cheat our manifold checks

#     flip_idx = np.where((mesh._halfedges['vertex'] == 3) & (mesh._halfedges['face'] == 0))[0][0]

#     mesh.edge_flip(flip_idx)

#     assert _test_normals(mesh)

# def test_double_edge_flip_topology():
#     vertices = _generate_vertices(4)
#     mesh = triangle_mesh.TriangleMesh(vertices, PRE_FLIP_FACES)
#     mesh._vertices['valence'][:] = 12  # cheat our manifold checks

#     flip_idx = np.where((mesh._halfedges['vertex'] == 3) & (mesh._halfedges['face'] == 0))[0][0]

#     mesh.edge_flip(flip_idx)
#     mesh._vertices['valence'][:] = 12  # cheat our manifold checks

#     flip_idx = np.where((mesh._halfedges['vertex'] == 2) & (mesh._halfedges['face'] == 1))[0][0]

#     mesh.edge_flip(flip_idx)

#     # flip_idx = np.where((mesh._halfedges['vertex'] == 0) & (mesh._halfedges['face'] == 0))[0][0]

#     # mesh.edge_flip(flip_idx)

#     # flip_idx = np.where((mesh._halfedges['vertex'] == 1) & (mesh._halfedges['face'] == 0))[0][0]

#     # mesh.edge_flip(flip_idx)

#     # flip_idx = np.where((mesh._halfedges['vertex'] == 2) & (mesh._halfedges['face'] == 0))[0][0]

#     # mesh.edge_flip(flip_idx)

#     assert _test_topology(mesh, PRE_FLIP_H_VERTEX, PRE_FLIP_H_FACE, PRE_FLIP_H_TWIN, PRE_FLIP_H_NEXT, PRE_FLIP_H_PREV)

def test_edge_collapse_topology():
    vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(vertices, K4_FACES)
    mesh._vertices['valence'][:] = 12  # cheat our manifold checks

    collapse_idx = np.where((mesh._halfedges['vertex'] == 2) & (mesh._halfedges['face'] == 1))[0][0]

    mesh.edge_collapse(collapse_idx)

    assert _test_topology(mesh, K4_COLLAPSED_H_VERTEX, K4_COLLAPSED_H_FACE, K4_COLLAPSED_H_TWIN, K4_COLLAPSED_H_NEXT, K4_COLLAPSED_H_PREV)

def test_edge_collapse_normals():
    vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(vertices, K4_FACES)

    collapse_idx = np.where((mesh._halfedges['vertex'] == 2) & (mesh._halfedges['face'] == 1))[0][0]

    mesh.edge_collapse(collapse_idx)

    assert _test_normals(mesh)

def test_edge_split_topology():
    vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(vertices, PRE_SPLIT_FACES)
    mesh._vertices['valence'][:] = 12  # cheat our manifold checks

    split_idx = np.where((mesh._halfedges['vertex'] == 3) & (mesh._halfedges['face'] == 0))[0][0]

    mesh.edge_split(split_idx)

    assert _test_topology(mesh, POST_SPLIT_H_VERTEX, POST_SPLIT_H_FACE, POST_SPLIT_H_TWIN, POST_SPLIT_H_NEXT, POST_SPLIT_H_PREV)

def test_edge_split_normals():
    vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(vertices, PRE_SPLIT_FACES)
    mesh._vertices['valence'][:] = 12  # cheat our manifold checks

    split_idx = np.where((mesh._halfedges['vertex'] == 3) & (mesh._halfedges['face'] == 0))[0][0]

    mesh.edge_split(split_idx)

    assert _test_normals(mesh)

# def test_split_collapse_topology():
#     vertices = _generate_vertices(4)
#     mesh = triangle_mesh.TriangleMesh(vertices, PRE_SPLIT_FACES)
#     mesh._vertices['valence'][:] = 12  # cheat our manifold checks

#     split_idx = np.where((mesh._halfedges['vertex'] == 3) & (mesh._halfedges['face'] == 0))[0][0]

#     mesh.edge_split(split_idx)

#     collapse_idx = np.where((mesh._halfedges['vertex'] == 3) & (mesh._halfedges['face'] == 3))[0][0]
#     mesh._vertices['valence'][:] = 12  # cheat our manifold checks
#     mesh.edge_collapse(collapse_idx)

#     assert _test_topology(mesh, PRE_SPLIT_H_VERTEX, PRE_SPLIT_H_FACE, PRE_SPLIT_H_TWIN, PRE_SPLIT_H_NEXT, PRE_SPLIT_H_PREV)


# def test_flip_split_topology():
#     vertices = _generate_vertices(4)
#     mesh = triangle_mesh.TriangleMesh(vertices, POST_FLIP_FACES[::-1])
#     mesh._vertices['valence'][:] = 12  # cheat our manifold checks
#     flip_idx = np.where((mesh._halfedges['vertex'] == 2) & (mesh._halfedges['face'] == 0))[0][0]

#     mesh.edge_flip(flip_idx)
#     mesh._vertices['valence'][:] = 12  # cheat our manifold checks

#     mesh.edge_split(flip_idx)

#     assert _test_topology(mesh, POST_SPLIT_H_VERTEX, POST_SPLIT_H_FACE, POST_SPLIT_H_TWIN, POST_SPLIT_H_NEXT, POST_SPLIT_H_PREV)


# def test_regularize():
#     vertices = _generate_vertices(5)
#     mesh = triangle_mesh.TriangleMesh(vertices, POST_SPLIT_FACES)

#     mesh.regularize()

#     assert np.all(mesh._valences <= 6)

# Replaced test_split_collapse_topology(), test_flip_split_topology(), test_regularize()
# with test_remesh(), which tests all of the elements in the previous three tests. The
# only problem is it's harder to pin down the error when something about this test breaks.
def test_remesh():
    tris = _generate_test_sphere()
    mesh = triangle_mesh.TriangleMesh.from_np_stl(tris)
    
    pre_manifold = mesh.manifold
    mesh.remesh()
    post_manifold = mesh.manifold

    assert (pre_manifold & post_manifold)

def test_resize():

    k4_vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(k4_vertices, K4_FACES)

    # Test a vector
    size = 10
    test_vec = np.zeros(size)
    test_vec = mesh._resize(test_vec)
    test_vec_true = np.all(test_vec[:size] == 0) & np.all(test_vec[size:] == -1)

    test_vec = np.zeros(size)
    test_vec[7:9] = -1
    test_vec_2 =  mesh._resize(test_vec)
    test_vec_true_2 = np.all(test_vec_2[:size] == test_vec) & np.all(test_vec[size:] == -1)

    # Test a 2D array along axis 0
    _vertices = mesh._vertices['position']
    _vertices = mesh._resize(_vertices)
    ax0_true = np.all(_vertices[0:4] == mesh.vertices) & np.all(_vertices[4:] == -1)

    _vertices = mesh._vertices['position']
    _vertices = mesh._resize(_vertices, axis=1)
    ax1_true = np.all(_vertices[:, 0:3] == mesh.vertices) & np.all(_vertices[:,3:] == -1)

    assert(test_vec_true & test_vec_true_2 & ax0_true & ax1_true)

def test_snap_faces():
    vertices = _generate_vertices(12)
    mesh = triangle_mesh.TriangleMesh(vertices, PRE_SNAP_FACES)

    _h0 = np.where((mesh._halfedges['vertex'] == 0) & (mesh._halfedges['face'] == 0))[0][0]
    _h1 = np.where((mesh._halfedges['vertex'] == 6) & (mesh._halfedges['face'] == 4))[0][0]

    mesh._snap_faces(_h0,_h1)

    assert _test_topology(mesh, POST_SNAP_H_VERTEX, POST_SNAP_H_FACE, POST_SNAP_H_TWIN, POST_SNAP_H_NEXT, POST_SNAP_H_PREV)

def test_load_save_stl():
    import os

    tris = _generate_test_sphere()
    mesh = triangle_mesh.TriangleMesh.from_np_stl(tris)
    test_fn = 'test_sphere.stl'
    mesh.to_stl(test_fn)
    mesh2 = triangle_mesh.TriangleMesh.from_stl(test_fn)

    os.remove(test_fn)

    # Weak check, but better than nothing
    np.testing.assert_array_almost_equal(mesh._vertices['position'], mesh2._vertices['position'])

def test_load_save_ply():
    import os

    tris = _generate_test_sphere()
    mesh = triangle_mesh.TriangleMesh.from_np_stl(tris)
    test_fn = 'test_sphere.ply'
    mesh.to_ply(test_fn)
    mesh2 = triangle_mesh.TriangleMesh.from_ply(test_fn)

    os.remove(test_fn)
    
    # Weak check, but better than nothing
    np.testing.assert_array_almost_equal(mesh._vertices['position'], mesh2._vertices['position'])
