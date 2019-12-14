import pytest
pytestmark = pytest.mark.xfail(reason="Tests are broken and need fixing")

import numpy as np

import PYME.experimental._octree as octree
from PYME.experimental import dual_marching_cubes

from PYME.experimental import triangular_mesh

def _generate_test_sphere():
    """
    Create a sphere for testing purposes.
    """
    # Generate a Gaussian point cloud to approximate a sphere
    density = 0.5
    voxelsize = 1e2
    n_points = int(density * 8 * (voxelsize * voxelsize * voxelsize))
    x = np.random.randn(n_points) * voxelsize/2
    y = np.random.randn(n_points) * voxelsize/2
    z = np.random.randn(n_points) * voxelsize/2

    # Discretize the points with an octree
    ot = octree.Octree([-3*voxelsize, 3*voxelsize, -3*voxelsize, 
                       3*voxelsize, -3*voxelsize, 3*voxelsize], 
                       maxdepth=3)
    ot.add_points(np.vstack((x,y,z)).astype('float32').T)

    # Construct a mesh over the octree
    dmc = dual_marching_cubes.DualMarchingCubes(density/100.)
    dmc.set_octree(ot)
    tris = dmc.march()

    return tris

def test_face_normal_magnitude():
    tris = _generate_test_sphere()
    mesh = triangular_mesh.TriangularMesh.from_np_stl(tris)
    fn = mesh.face_normals
    fnn = np.linalg.norm(fn, axis=1)
    y = np.ones(fnn.shape)

    np.testing.assert_array_almost_equal(fnn, y, 4)

def test_vertex_normal_magnitude():
    tris = _generate_test_sphere()
    mesh = triangular_mesh.TriangularMesh.from_np_stl(tris)
    vn = mesh.vertex_normals
    vnn = np.linalg.norm(vn, axis=1)
    y = np.ones(vnn.shape)

    np.testing.assert_array_almost_equal(vnn, y, 4)

def test_vertex_normal_sign():
    tris = _generate_test_sphere()
    mesh = triangular_mesh.TriangularMesh.from_np_stl(tris)
    vn = mesh.vertex_normals
    # Since this is a sphere centered on the origin, the normals
    # at a vertex should be roughly equal to their normalized 
    # position vectors (and they are). We settle for checking that
    # the sign of each component is correct.
    vnn = np.linalg.norm(mesh.vertices, axis=1)
    vnnn = mesh.vertices/np.vstack([vnn, vnn, vnn]).T

    np.testing.assert_array_almost_equal(np.sign(vn), np.sign(vnnn), 4)

def test_add_vertex_location():
    # Construct a single triangle
    vertices = np.array([[0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]])
    faces = np.array([[0, 1, 2]])
    mesh = triangular_mesh.TriangularMesh(vertices, faces)
    mesh.add_vertex(0)

    np.testing.assert_array_almost_equal(np.array([1./3, 1./3, 0.5]),
                                         mesh.vertices[-1, :], 4)

def test_add_vertex_n_faces():
    # Construct a single triangle
    vertices = np.array([[0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]])
    faces = np.array([[0, 1, 2]])
    mesh = triangular_mesh.TriangularMesh(vertices, faces)
    mesh.add_vertex(0)
    
    assert ((mesh.faces.shape == (3, 3)) and (mesh.vertices.shape == (4,3)))

def test_add_vertex_faces():
    # Construct a single triangle
    vertices = np.array([[0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]])
    faces = np.array([[0, 1, 2]])
    mesh = triangular_mesh.TriangularMesh(vertices, faces)
    mesh.add_vertex(0)

    expected_faces = np.array([[0,3,2],[0,1,3],[1,2,3]])

    np.testing.assert_array_almost_equal(mesh.faces, expected_faces, 4)

def test_add_vertex_face_normals():
    # Construct a single triangle
    vertices = np.array([[0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]])
    faces = np.array([[0, 1, 2]])
    mesh = triangular_mesh.TriangularMesh(vertices, faces)
    original_normal = mesh.face_normals
    mesh.add_vertex(0)

    assert(np.all(np.sign(mesh.face_normals*original_normal[None,:])>=0))

def test_neighbors():
    # Construct a single triangle
    vertices = np.array([[0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]])
    faces = np.array([[0, 1, 2]])
    mesh = triangular_mesh.TriangularMesh(vertices, faces)

    # Calculated neighbors
    n0 = mesh.neighbors[0]
    n1 = mesh.neighbors[1]
    n2 = mesh.neighbors[2]

    # Expected neighbors
    en0 = [1, 2]
    en1 = [0, 2]
    en2 = [0, 1]

    assert((n0 == en0) and (n1 == en1) and (n2 == en2))

def test_face_indices():
    # Construct two triangles
    vertices = np.array([[0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5],
                        [1./3, 1./3, 0.5]])
    faces = np.array([[0,1,2],[0,3,2],[0,1,3],[1,2,3]])
    mesh = triangular_mesh.TriangularMesh(vertices, faces)

    expected_face_indices = [[0,1,2],[0,2,3],[0,1,3],[1,2,3]]

    assert(mesh.face_indices == expected_face_indices)