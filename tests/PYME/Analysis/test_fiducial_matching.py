import numpy as np
import pytest

def test_match_points():
    from PYME.Analysis import fiducial_matching
    x1 = 1000*np.random.rand(100, 2)
    #print(x1.shape)
    
    # add some noise 
    x2 = x1 + np.random.randn(100, 2)
    # and discard some points
    mask = np.random.rand(100) < 0.9
    x2 = x2[mask, :]

    idx, score = fiducial_matching.match_points(x1, x2, scale=10, gui=False)

    print(mask, idx, score)
    print(idx[mask])

    #print(x1[mask, :]- x2[idx, :][mask, :])

    #assert(np.allclose(x1[mask, :], x2[idx, :][mask, :], atol=3))

    assert(np.allclose(idx[mask], np.arange(mask.sum())))
    #assert(False)

@pytest.mark.xfail(reason="matching under rotation is still not robust")
def test_match_points_rotation():
    from PYME.Analysis import fiducial_matching
    x1 = 10000*np.random.rand(100, 2)
    #print(x1.shape)
    
    # add some noise 
    x2 = x1 + np.random.randn(100, 2)
    # and discard some points
    mask = np.random.rand(100) < 0.9
    x2_ = x2[mask, :]

    #add rotation
    t = np.pi/64
    #t = 0
    x2 = np.dot(x2_, np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]))

    idx, score = fiducial_matching.match_points(x1, x2, scale=10, gui=False)

    print(mask, idx, score)
    print(idx[mask])

    #print(x1[mask, :]- x2[idx, :][mask, :])

    #assert(np.allclose(x1[mask, :], x2[idx, :][mask, :], atol=3))

    assert(np.allclose(idx[mask], np.arange(mask.sum())))