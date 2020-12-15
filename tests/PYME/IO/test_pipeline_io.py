import pytest

def test_hdf_import():
    import os
    import numpy as np
    from PYME import resources
    from PYME.LMVis import pipeline
    
    
    p = pipeline.Pipeline(filename=os.path.join(resources.get_test_data_dir(), 'test_hdf.hdf'))
    
    # did we load the correct number of localisations?
    assert (len(p['x']) == 4281)

    # did we correctly estimate the image bounds?
    assert np.allclose(p.imageBounds.bounds, (-686.9195, -597.1356, 80.27731, 47.10489, -265.4904, 232.2955))

@pytest.mark.xfail(reason='column parsing currently performed in GUI code')
def test_csv_import():
    import os
    import numpy as np
    from PYME import resources
    from PYME.LMVis import pipeline
    
    p = pipeline.Pipeline(filename=os.path.join(resources.get_test_data_dir(), 'test_csv.csv'))

    # did we load the correct number of localisations?
    assert (len(p['x']) == 4281)

    # did we correctly estimate the image bounds?
    assert np.allclose(p.imageBounds.bounds, (-686.9195, -597.1356, 80.27731, 47.10489, -265.4904, 232.2955))
