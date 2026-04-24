
import numpy as np
from PYME.IO.image import ImageStack
from PYME.IO.DataSources.CropDataSource import DataSource


def _make_datasource(nx=8, ny=6, nz=3, nt=2, nc=4):
    """5D datasource where channel c has constant pixel value (c+1)*10."""
    data = np.zeros((nx, ny, nz, nt, nc), dtype='float32')
    for c in range(nc):
        data[:, :, :, :, c] = (c + 1) * 10.0
    return ImageStack(data).data_xyztc


def test_crange_shape():
    """Extracting a channel range produces the correct output shape."""
    ds = _make_datasource(nx=8, ny=6, nz=3, nt=2, nc=4)
    cropped = DataSource(ds, crange=(1, 3))
    assert cropped.shape == (8, 6, 3, 2, 2)


def test_crange_data_values():
    """Each extracted channel contains the correct pixel values."""
    ds = _make_datasource(nc=4)
    for chan in range(4):
        cropped = DataSource(ds, crange=(chan, chan + 1))
        out = cropped[:, :, :, :, :]
        assert np.allclose(out, (chan + 1) * 10.0), f"Wrong values for channel {chan}"


def test_crange_combined_with_other_dims():
    """crange works correctly alongside zrange and trange."""
    ds = _make_datasource(nz=3, nt=2, nc=4)
    cropped = DataSource(ds, zrange=(0, 2), trange=(1, 2), crange=(2, 3))
    assert cropped.shape[2:] == (2, 1, 1)
    assert np.allclose(cropped[:, :, :, :, :], 30.0)
