def test_coalesce():
    from PYME.Analysis.points import multiview
    import numpy as np
    assigned = np.array([0,0,1,1,2,3,4,5,5,5,6], 'i')
    x = np.random.rand(len(assigned))
    x_out = multiview.coalesce_dict_sorted({'x':x}, assigned, ['x',], {})['x']
    
    for j in np.unique(assigned):
        assert np.allclose(x_out[j] , np.mean(x[assigned==j]))
        
    assert len(x_out) == assigned.max() + 1


def test_coalesce_incomplete():
    from PYME.Analysis.points import multiview
    import numpy as np
    assigned = np.array([1, 1, 2, 3, 5, 5, 5, 6, 9,9], 'i')
    x = np.random.rand(len(assigned))
    x_out = multiview.coalesce_dict_sorted({'x': x}, assigned, ['x', ], {})['x']
    
    for j in np.unique(assigned):
        assert np.allclose(x_out[j] , np.mean(x[assigned == j]))
    
    assert len(x_out) == assigned.max() + 1


def test_coalesce_nontrivial():
    from PYME.Analysis.points import multiview
    import numpy as np
    assigned = np.array([0, 0, 1, 1, 2, 3, 4, 5, 5, 5, 6], 'i')
    x = np.random.rand(len(assigned))
    x_out = multiview.coalesce_dict_sorted({'x': x}, assigned, ['x', ], {}, discard_trivial=True)['x']
    
    assert len(x_out) == len(np.unique(assigned[assigned >=1]))
    
def test_coalesce_incomplete_nontrivial():
    from PYME.Analysis.points import multiview
    import numpy as np
    assigned = np.array([1, 1, 2, 3, 5, 5, 5, 6, 9,9], 'i')
    x = np.random.rand(len(assigned))
    x_out = multiview.coalesce_dict_sorted({'x': x}, assigned, ['x', ], {}, discard_trivial=True)['x']

    assert len(x_out) == len(np.unique(assigned[assigned >= 1]))
    
    
def test_clumping():
    import numpy as np
    from PYME.Analysis.points.DeClump import findClumps
    
    x = y = np.zeros(5, 'f')
    dx = np.ones(5, 'f')
    t = np.array([0,0,1,3,6]).astype('i')
    
    assert np.allclose(findClumps(t+9, x, y, dx, 0, True), [1, 1, 1, 2, 3])
    assert np.allclose(findClumps(t+9, x, y, dx, 1, True), [1, 1, 1, 1, 2])
    assert np.allclose(findClumps(t+9, x, y, dx, 2, True), [1, 1, 1, 1, 1])
    assert np.allclose(findClumps(t + 9, x, y, dx, -1, True), [1, 1, 2, 3, 4])
    
    assert np.allclose(findClumps(t, x, y, dx,0, True), [1,1,1,2,3])
    assert np.allclose(findClumps(t, x, y, dx, 1, True), [1, 1, 1, 1, 2])
    assert np.allclose(findClumps(t, x, y, dx, 2, True), [1, 1, 1, 1, 1])
    assert np.allclose(findClumps(t, x, y, dx, -1, True), [1, 1, 2, 3, 4])

    assert np.allclose(findClumps(t+1, x, y, dx, 0, False), [1, 2, 1, 3, 4])
    assert np.allclose(findClumps(t+1, x, y, dx, 1, False), [1, 2, 1, 1, 3])
    assert np.allclose(findClumps(t+1, x, y, dx, 2, False), [1, 2, 1, 1, 1])
    assert np.allclose(findClumps(t+1, x, y, dx, -1, False), [1, 2, 3, 4, 5])
    
    assert np.allclose(findClumps(t, x, y, dx, 0, False), [1, 2, 1, 3, 4])
    assert np.allclose(findClumps(t, x, y, dx, 1, False), [1, 2, 1, 1, 3])
    assert np.allclose(findClumps(t, x, y, dx, 2, False), [1, 2, 1, 1, 1])
    assert np.allclose(findClumps(t, x, y, dx, -1, False), [1, 2, 3, 4, 5])