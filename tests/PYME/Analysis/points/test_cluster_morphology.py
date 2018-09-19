
import numpy as np
from PYME.Analysis.points import cluster_morphology

# cube
x_cube = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=float)
y_cube = np.array([1, 1, -1, 1, -1, 1, -1, -1], dtype=float)
z_cube = np.array([1, -1, 1, 1, 1, -1, -1, -1], dtype=float)

def test_anisotropy():
    x = np.arange(10)
    # perfectly anisotropic
    output = cluster_morphology.measure_3d(x, x, x)
    np.testing.assert_almost_equal(output['anisotropy'][0], 1.)

    # isotropic
    output = cluster_morphology.measure_3d(x_cube, y_cube, z_cube)
    np.testing.assert_almost_equal(output['anisotropy'][0], 0.)

def test_principle_axes():
    x = np.arange(10)
    nill = np.zeros_like(x)
    # perfectly anisotropic
    output = cluster_morphology.measure_3d(x, nill, nill)
    np.testing.assert_array_almost_equal(np.array([output['sigma0'][0], output['sigma1'][0], output['sigma2'][0]]),
                                         np.array([np.std(x, ddof=1), 0, 0]))

    np.testing.assert_almost_equal(np.array([1, 0, 0]), output['axis0'][0])
