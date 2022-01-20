
import numpy as np
from PYME.Analysis.points import spherical_harmonics
from PYME.Analysis.points import coordinate_tools
from skimage import morphology
from pytest import mark

# Make a shell
R_SPHERE = 49.
ball = morphology.ball(R_SPHERE, dtype=int)
SPHERICAL_SHELL = morphology.binary_dilation(ball) - ball
X, Y, Z = np.where(SPHERICAL_SHELL)
X = X.astype(float)
Y = Y.astype(float)
Z = Z.astype(float)
X_C, Y_C, Z_C = X - R_SPHERE, Y - R_SPHERE, Z - R_SPHERE
AZIMUTH, ZENITH, R = coordinate_tools.cartesian_to_spherical(X_C, Y_C, Z_C)
R = np.sqrt(X_C**2 + Y_C**2 + Z_C**2)
FITTER = spherical_harmonics.ScaledShell()
FITTER.set_fitting_points(X, Y, Z)
FITTER.fit_shell()

def test_scaled_fitter():
    x, y, z = FITTER.get_fitted_shell(AZIMUTH, ZENITH)
    # check we get back what we put in, but note that we put in ints on a 3D grid, so allow for sqrt(3) mismatch + precision
    np.testing.assert_allclose(np.sort(x), np.sort(X), rtol=0.05, atol=np.sqrt(3))
    np.testing.assert_allclose(np.sort(y), np.sort(Y), rtol=0.05, atol=np.sqrt(3))
    np.testing.assert_allclose(np.sort(z), np.sort(Z), rtol=0.05, atol=np.sqrt(3))

def test_distance_to_shell():
    query = ([FITTER.x0], [FITTER.y0], [FITTER.z0])
    distance, closest_point = FITTER.distance_to_shell(query, 0.05)
    np.testing.assert_almost_equal(R_SPHERE, distance[0], decimal=0)

    x_q = np.arange(R_SPHERE, 2 * R_SPHERE)
    y_q = R_SPHERE * np.ones_like(x_q)
    z_q = y_q
    query = (x_q, y_q, z_q)
    distance, closest_points = FITTER.distance_to_shell(query, 0.05)
    assert np.abs(distance).min() / R_SPHERE < 0.05
    # everything after the first couple should be the same,
    x_closest_short = closest_points[0][10:]
    # x should be x0 + radius
    np.testing.assert_allclose(x_closest_short, FITTER.x0 + R_SPHERE, atol=np.sqrt(3))

    y_closest_short, z_closest_short = closest_points[1][10:], closest_points[2][10:]
    # y, z, should be roughly y0, z0
    np.testing.assert_allclose(y_closest_short, FITTER.y0, atol=np.sqrt(3))
    np.testing.assert_allclose(z_closest_short, FITTER.z0, atol=np.sqrt(3))

def test_approximate_normal():
    # test x
    normal = FITTER.approximate_normal(FITTER.x0 + R_SPHERE, FITTER.y0, FITTER.z0)
    np.testing.assert_almost_equal(normal, np.array([1, 0, 0]), decimal=3)
    # test y
    normal = FITTER.approximate_normal(FITTER.x0, FITTER.y0 + R_SPHERE, FITTER.z0)
    np.testing.assert_almost_equal(normal, np.array([0, 1, 0]), decimal=3)
    # test y+z
    normal = FITTER.approximate_normal(FITTER.x0, FITTER.y0 + R_SPHERE, FITTER.z0 + R_SPHERE)
    np.testing.assert_almost_equal(normal, np.array([0, 1, 1]) / np.sqrt(2), decimal=3)

@mark.xfail
def test_approximate_normal_small_angles():
    # test z
    normal = FITTER.approximate_normal(FITTER.x0, FITTER.y0, FITTER.z0 + R_SPHERE)
    np.testing.assert_almost_equal(normal, np.array([0, 0, 1]), decimal=3)

def test_vectorized_approximate_normal():
    # test vectorized input, all together
    query = np.stack([
        [FITTER.x0 + R_SPHERE, FITTER.y0, FITTER.z0],
        [FITTER.x0, FITTER.y0 + R_SPHERE, FITTER.z0],
        # [FITTER.x0, FITTER.y0, FITTER.z0 + R_SPHERE],
        [FITTER.x0, FITTER.y0 + R_SPHERE, FITTER.z0 + R_SPHERE]
    ])
    normals = FITTER.approximate_normal(query[:, 0], query[:, 1], query[:, 2])
    true_normals = np.stack([
        [1, 0, 0],
        [0, 1, 0],
        # [0, 0, 1],
        [0, 1./np.sqrt(2), 1./np.sqrt(2)]
    ])
    np.testing.assert_almost_equal(normals, true_normals, decimal=3)

def test_distance_to_shell_along_vector_from_point():
    up = np.array([0, 0, 1])

    query_point = [FITTER.x0 + (0.2 * R_SPHERE), FITTER.y0, FITTER.z0]
    dist = FITTER.distance_to_shell_along_vector_from_point(up, query_point, np.arange(1, 50))
    ground_truth = np.sqrt((R_SPHERE ** 2) - ((0.2 * R_SPHERE + FITTER.x0 - FITTER.x0) ** 2))
    np.testing.assert_almost_equal(ground_truth, dist, decimal=0)
    print(ground_truth - dist)

    query_point = [FITTER.x0 + (0.5 * R_SPHERE), FITTER.y0, FITTER.z0]
    dist = FITTER.distance_to_shell_along_vector_from_point(up, query_point, np.arange(1, 50))
    ground_truth = np.sqrt((R_SPHERE ** 2) - ((0.5 * R_SPHERE + FITTER.x0 - FITTER.x0) ** 2))
    np.testing.assert_almost_equal(ground_truth, dist, decimal=0)

    query_point = [FITTER.x0 + (0.75 * R_SPHERE), FITTER.y0, FITTER.z0]
    dist = FITTER.distance_to_shell_along_vector_from_point(up, query_point, np.arange(1, 50))
    ground_truth = np.sqrt((R_SPHERE ** 2) - ((0.75 * R_SPHERE + FITTER.x0 - FITTER.x0) ** 2))
    np.testing.assert_almost_equal(ground_truth, dist, decimal=0)

    query_point = [FITTER.x0 + (0.9 * R_SPHERE), FITTER.y0, FITTER.z0]
    dist = FITTER.distance_to_shell_along_vector_from_point(up, query_point, np.arange(1, 50))
    ground_truth = np.sqrt((R_SPHERE ** 2) - ((0.9 * R_SPHERE + FITTER.x0 - FITTER.x0) ** 2))
    np.testing.assert_almost_equal(ground_truth, dist, decimal=0)

def test_distance_to_shell_along_vector_from_points():
    up = np.array([0, 0, 1])
    xq = FITTER.x0 + R_SPHERE * np.array([0.25, 0.5, 0.75, 0.9])
    yq = FITTER.y0 * np.ones_like(xq)
    zq = FITTER.z0 * np.ones_like(xq)
    query = np.stack([xq, yq, zq]).T
    guess = [FITTER._find_guess_for_distance_to_shell_along_vector_from_point(up, query[gi], np.arange(1, 50)) for gi in range(len(query))]
    dist = FITTER.distance_to_shell_along_vector_from_point(up, query, guess)
    ground_truth = np.sqrt((R_SPHERE ** 2) - ((xq - FITTER.x0) ** 2))
    np.testing.assert_array_almost_equal(ground_truth, dist, decimal=0)

def test_density_estimate():
    from PYME.recipes.surface_fitting import SHShellRadiusDensityEstimate

    hist_out = SHShellRadiusDensityEstimate(sampling_nm=[1, 1, 1],
                                            jitter_iterations=5).apply_simple(FITTER)
    r2norm = hist_out['counts'] / (hist_out['bin_centers']**2)
    rel = r2norm / r2norm[-1]
    # drop the lowest bin and check that we're within 5% of the rest
    assert np.all(np.abs(rel[1:] - 1) < 0.05)
    v_sphere = (4/3 * np.pi * R_SPHERE**3)  # [nm^3]
    v_sphere_out_nm3 = hist_out.mdh['SHShellRadiusDensityEstimate.Volume'] * 1e9  # [nm^3]
    assert abs(v_sphere - v_sphere_out_nm3) / v_sphere < 0.05
