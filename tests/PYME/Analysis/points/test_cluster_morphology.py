
import numpy as np
from PYME.Analysis.points import cluster_morphology

def test_relative_anisotropy():
    x = np.arange(10)
    # perfectly anisotropic
    output = cluster_morphology.measure_3d(x, x, x)
    np.testing.assert_almost_equal(output['anisotropy'][0], 1.)

    # isotropic, using a cube
    x = np.array([1, 1, 1, -1, -1, -1, 1, -1], dtype=float)
    y = np.array([1, 1, -1, 1, -1, 1, -1, -1], dtype=float)
    z = np.array([1, -1, 1, 1, 1, -1, -1, -1], dtype=float)
    output = cluster_morphology.measure_3d(x, y, z)
    np.testing.assert_almost_equal(output['anisotropy'][0], 0.)