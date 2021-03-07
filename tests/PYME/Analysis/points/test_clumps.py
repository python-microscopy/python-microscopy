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

def test_pydeclump_no_gap():
    from PYME.Analysis.points.DeClump import pyDeClump
    import numpy as np
    x = np.array([0, 0])
    y = x
    t = np.array([0, 0], dtype=np.int32)
    dist = 2 * np.ones(2)
    assigned = pyDeClump.findClumps(t.astype(np.int32), x, y, dist, 0)
    assert np.all(assigned == 1)

def test_pydeclump_gap():
    from PYME.Analysis.points.DeClump import pyDeClump
    import numpy as np
    x = np.array([0, 0, 0])
    y = x
    t = np.array([0, 1, 3], dtype=np.int32)
    dist = 2 * np.ones(2)
    assigned = pyDeClump.findClumps(t.astype(np.int32), x, y, dist, 1)
    np.testing.assert_array_equal([1, 1, 2], assigned)

def test_declump_no_gap():
    from PYME.Analysis.points.DeClump import deClump
    import numpy as np
    x = np.array([0, 0])
    y = x
    t = np.array([0, 0], dtype=np.int32)
    dist = 2 * np.ones(2)
    assigned = deClump.findClumps(t.astype(np.int32), x.astype(np.float32), y.astype(np.float32), dist.astype(np.float32), 0)
    assert np.all(assigned == 1)

def test_declump_gap():
    from PYME.Analysis.points.DeClump import deClump
    import numpy as np
    x = np.array([0, 0, 0])
    y = x
    t = np.array([0, 1, 3], dtype=np.int32)
    dist = 2 * np.ones(2)
    assigned = deClump.findClumps(t.astype(np.int32), x.astype(np.float32), y.astype(np.float32), dist.astype(np.float32), 1)
    np.testing.assert_array_equal([1, 1, 2], assigned)
