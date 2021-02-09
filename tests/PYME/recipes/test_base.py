
from PYME.recipes import base

def test_crop():
    from PYME.IO.DataSources.ArrayDataSource import ArrayDataSource
    from PYME.IO.image import ImageStack
    import numpy as np
    d = np.arange(10) * np.ones((5, 4))[:, :, None]
    im = ImageStack(data=ArrayDataSource(d), haveGUI=False)
    
    # test null crop
    out = base.Crop().apply_simple(im)
    np.testing.assert_array_equal(d.shape, out.data.shape[:-1])
    
    # test x crop
    out = base.Crop(x_range=[1, d.shape[0] - 1]).apply_simple(im)
    assert out.data.shape[0] == d.shape[0] - 2
    np.testing.assert_array_equal(d.shape[1:], out.data.shape[1:-1])

    # test z/t crop
    out = base.Crop(t_range=[3, d.shape[2] - 3]).apply_simple(im)
    assert out.data.shape[2] == d.shape[2] - 6
    np.testing.assert_array_equal(d.shape[:2], out.data.shape[:2])

def test_bad_crop_catch():
    from PYME.IO.DataSources.ArrayDataSource import ArrayDataSource
    from PYME.IO.image import ImageStack
    import numpy as np
    d = np.arange(10) * np.ones((5, 4))[:, :, None]
    im = ImageStack(data=ArrayDataSource(d), haveGUI=False)
    
    try:
        out = base.Crop(t_range=[7, d.shape[2] - 7]).apply_simple(im)
    except AssertionError:
        pass
    else:
        raise ValueError
