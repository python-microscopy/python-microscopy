import numpy as np

from PYME.experimental import marching_cubes

from PYME.experimental import triangle_mesh

# Ground truth faces and halfedges for a K3 planar graph
K3_FACES = np.array([[2,0,1]])
K3_H_VERTEX = np.array([0,1,2])
K3_H_FACE = np.array([0,0,0])
K3_H_TWIN = np.array([-1,-1,-1])
K3_H_NEXT = np.array([1,2,0])
K3_H_PREV = np.array([2,0,1])
K3_NIGHBORS = np.array([[1,2],[0,2],[0,1]])

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
    vm = (_vertex[:,None] == mesh._h_vertex[None, :])
    fm = (_face[:,None] == mesh._h_face[None, :])
    m = np.argwhere(vm & fm)[:, 1]

    t_bool = (mesh._h_twin[m] == -1) & (_twin == -1)

    twin_vert_eq = (mesh._h_vertex[mesh._h_twin][m] == _vertex[_twin]) | t_bool
    twin_face_eq = (mesh._h_face[mesh._h_twin][m] == _face[_twin]) | t_bool
    next_vert_eq = (mesh._h_vertex[mesh._h_next][m] == _vertex[_next])
    next_face_eq = (mesh._h_face[mesh._h_next][m] == _face[_next])
    prev_vert_eq = (mesh._h_vertex[mesh._h_prev][m] == _vertex[_prev])
    prev_face_eq = (mesh._h_face[mesh._h_prev][m] == _face[_prev])

    return (np.all(twin_vert_eq & twin_face_eq & next_vert_eq & next_face_eq & prev_vert_eq & prev_face_eq) & _test_halfedges(mesh))

def _test_halfedges(mesh):
    # Check that the next next vertex is the same as the prev vertex
    next_next_prev = (mesh._h_vertex[mesh._h_next[mesh._h_next[mesh._vertex_halfedges]]] == mesh._h_vertex[mesh._h_prev[mesh._vertex_halfedges]])

    # Check that the prev vertex is the same as the twin vertex
    twins = mesh._h_twin[mesh._vertex_halfedges]
    t_idx = (twins != -1)
    twin_prev = (mesh._h_vertex[twins[t_idx]] == mesh._h_vertex[(mesh._h_prev[mesh._vertex_halfedges])[t_idx]])
    
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
    vn = mesh.vertex_normals
    # Since this is a sphere centered on the origin, the normals
    # at a vertex should be roughly equal to their normalized 
    # position vectors (and they are). We settle for checking that
    # the sign of each component is correct.
    vnn = np.linalg.norm(mesh.vertices, axis=1)
    vnnn = mesh.vertices/vnn[:, None]

    np.testing.assert_array_almost_equal(np.sign(vn), np.sign(vnnn))

def test_get_halfedges():

    k4_vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(k4_vertices, K4_FACES)

    assert _test_topology(mesh, K4_H_VERTEX, K4_H_FACE, K4_H_TWIN, K4_H_NEXT, K4_H_PREV)

def test_vertex_neighbors():

    k4_vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(k4_vertices, K4_FACES, 6)

    assert np.all(np.sort(mesh._h_vertex[mesh.vertex_neighbors[:,:K4_NEIGHBORS.shape[1]]], axis=1) == np.sort(K4_NEIGHBORS, axis=1))

def test_euler_characteristic():

    k4_vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(k4_vertices, K4_FACES)

    assert ((len(mesh.vertices) - len(mesh._h_vertex)/2. + len(mesh._faces)) == 2)

def test_missing_halfedges():

    k3_vertices = _generate_vertices(3)
    mesh = triangle_mesh.TriangleMesh(k3_vertices, K3_FACES)

    assert _test_topology(mesh, K3_H_VERTEX, K3_H_FACE, K3_H_TWIN, K3_H_NEXT, K3_H_PREV)

def test_edge_flip_topology():

    vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(vertices, PRE_FLIP_FACES)

    flip_idx = np.where((mesh._h_vertex == 3) & (mesh._h_face == 0))[0][0]

    mesh.edge_flip(flip_idx)

    assert _test_topology(mesh, POST_FLIP_H_VERTEX, POST_FLIP_H_FACE, POST_FLIP_H_TWIN, POST_FLIP_H_NEXT, POST_FLIP_H_PREV)

def test_edge_flip_normals():

    vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(vertices, PRE_FLIP_FACES)

    flip_idx = np.where((mesh._h_vertex == 3) & (mesh._h_face == 0))[0][0]

    mesh.edge_flip(flip_idx)

    assert _test_normals(mesh)

def test_double_edge_flip_topology():
    vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(vertices, PRE_FLIP_FACES)

    flip_idx = np.where((mesh._h_vertex == 3) & (mesh._h_face == 0))[0][0]

    mesh.edge_flip(flip_idx)

    flip_idx = np.where((mesh._h_vertex == 2) & (mesh._h_face == 1))[0][0]

    mesh.edge_flip(flip_idx)

    # flip_idx = np.where((mesh._h_vertex == 0) & (mesh._h_face == 0))[0][0]

    # mesh.edge_flip(flip_idx)

    # flip_idx = np.where((mesh._h_vertex == 1) & (mesh._h_face == 0))[0][0]

    # mesh.edge_flip(flip_idx)

    # flip_idx = np.where((mesh._h_vertex == 2) & (mesh._h_face == 0))[0][0]

    # mesh.edge_flip(flip_idx)

    assert _test_topology(mesh, PRE_FLIP_H_VERTEX, PRE_FLIP_H_FACE, PRE_FLIP_H_TWIN, PRE_FLIP_H_NEXT, PRE_FLIP_H_PREV)

def test_edge_collapse_topology():
    vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(vertices, K4_FACES)

    collapse_idx = np.where((mesh._h_vertex == 2) & (mesh._h_face == 1))[0][0]

    mesh.edge_collapse(collapse_idx)

    assert _test_topology(mesh, K4_COLLAPSED_H_VERTEX, K4_COLLAPSED_H_FACE, K4_COLLAPSED_H_TWIN, K4_COLLAPSED_H_NEXT, K4_COLLAPSED_H_PREV)

def test_edge_collapse_normals():
    vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(vertices, K4_FACES)

    collapse_idx = np.where((mesh._h_vertex == 2) & (mesh._h_face == 1))[0][0]

    mesh.edge_collapse(collapse_idx)

    assert _test_normals(mesh)

def test_edge_split_topology():
    vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(vertices, PRE_SPLIT_FACES)

    split_idx = np.where((mesh._h_vertex == 3) & (mesh._h_face == 0))[0][0]

    mesh.edge_split(split_idx)

    assert _test_topology(mesh, POST_SPLIT_H_VERTEX, POST_SPLIT_H_FACE, POST_SPLIT_H_TWIN, POST_SPLIT_H_NEXT, POST_SPLIT_H_PREV)

def test_edge_split_normals():
    vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(vertices, PRE_SPLIT_FACES)

    split_idx = np.where((mesh._h_vertex == 3) & (mesh._h_face == 0))[0][0]

    mesh.edge_split(split_idx)

    assert _test_normals(mesh)

def test_split_collapse_topology():
    vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(vertices, PRE_SPLIT_FACES)

    split_idx = np.where((mesh._h_vertex == 3) & (mesh._h_face == 0))[0][0]

    mesh.edge_split(split_idx)

    collapse_idx = np.where((mesh._h_vertex == 3) & (mesh._h_face == 3))[0][0]

    mesh.edge_collapse(collapse_idx)

    assert _test_topology(mesh, PRE_SPLIT_H_VERTEX, PRE_SPLIT_H_FACE, PRE_SPLIT_H_TWIN, PRE_SPLIT_H_NEXT, PRE_SPLIT_H_PREV)


def test_flip_split_topology():
    vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(vertices, POST_FLIP_FACES[::-1])

    flip_idx = np.where((mesh._h_vertex == 2) & (mesh._h_face == 0))[0][0]

    mesh.edge_flip(flip_idx)

    mesh.edge_split(flip_idx)

    assert _test_topology(mesh, POST_SPLIT_H_VERTEX, POST_SPLIT_H_FACE, POST_SPLIT_H_TWIN, POST_SPLIT_H_NEXT, POST_SPLIT_H_PREV)


def test_regularize():
    vertices = _generate_vertices(5)
    mesh = triangle_mesh.TriangleMesh(vertices, POST_SPLIT_FACES)

    mesh.regularize()

    assert np.all(mesh._valences <= 6)

def test_resize():

    k4_vertices = _generate_vertices(4)
    mesh = triangle_mesh.TriangleMesh(k4_vertices, K4_FACES, 10)

    # Test a vector
    size = 10
    test_vec = np.zeros(size)
    test_vec = mesh._resize(test_vec)
    test_vec_true = np.all(test_vec[:size] == 0) & np.all(test_vec[size:] == -1)

    test_vec = np.zeros(size)
    test_vec[7:9] = -1
    test_vec =  mesh._resize(test_vec)
    test_vec_true_2 = np.all(test_vec[:(size-2)] == 0) & np.all(test_vec[(size-2):] == -1)

    # Test a 2D array along axis 0
    _vertices = mesh._vertices['position']
    _vertices = mesh._resize(_vertices)
    ax0_true = np.all(_vertices[0:4] == mesh.vertices) & np.all(_vertices[4:] == -1)

    _vertices = mesh._vertices['position']
    _vertices = mesh._resize(_vertices, axis=1)
    ax1_true = np.all(_vertices[:, 0:3] == mesh.vertices) & np.all(_vertices[:,3:] == -1)

    assert(test_vec_true & test_vec_true_2 & ax0_true & ax1_true)