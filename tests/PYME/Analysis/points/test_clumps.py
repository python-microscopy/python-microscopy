def test_coalesce():
    from PYME.Analysis.points import multiview
    import numpy as np
    assigned = np.array([0,0,1,1,2,3,4,5,5,5,6], 'i')
    x = np.random.rand(len(assigned))
    x_out = multiview.coalesce_dict_sorted({'x':x}, assigned, ['x',], {})['x']
    
    for j in np.unique(assigned):
        assert (x_out[j] == np.mean(x[assigned==j]))
        
    assert len(x_out) == assigned.max() + 1


def test_coalesce_incomplete():
    from PYME.Analysis.points import multiview
    import numpy as np
    assigned = np.array([1, 1, 2, 3, 5, 5, 5, 6, 9,9], 'i')
    x = np.random.rand(len(assigned))
    x_out = multiview.coalesce_dict_sorted({'x': x}, assigned, ['x', ], {})['x']
    
    for j in np.unique(assigned):
        assert (x_out[j] == np.mean(x[assigned == j]))
    
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