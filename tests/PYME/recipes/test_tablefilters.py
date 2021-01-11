
from PYME.recipes import tablefilters

def test_random_selection():
    from PYME.IO import tabular
    import numpy as np
    d = tabular.DictSource({'test': np.arange(100)})

    out = tablefilters.RandomSubset(num_to_select=5,
                                    require_at_least_n=True).apply_simple(input=d)
    assert len(out) == 5
    
    out = tablefilters.RandomSubset(num_to_select=150,
                                    require_at_least_n=False).apply_simple(input=d)
    assert len(out) == 100
